import os
import json
import sqlite3
import asyncio
import time
import re
from datetime import datetime
from collections import Counter

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters,
)
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]
DB_PATH        = "jobs.db"

client = genai.Client(api_key=GEMINI_KEY)
MODEL  = "gemini-2.5-flash"

# ── Markdown escaping ─────────────────────────────────────────────────────────
# FIX #5: escape LLM output before injecting into Markdown templates
_MD_SPECIAL = re.compile(r'([*_`\[\]()~>#+=|{}.!\\-])')

def esc(text: str) -> str:
    """Escape special chars for Telegram MarkdownV2."""
    return _MD_SPECIAL.sub(r'\\\1', str(text)) if text else ""

# ── Database ──────────────────────────────────────────────────────────────────
# FIX #7: check_same_thread=False so to_thread() workers can also open connections
def _con():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = _con()
    con.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            company     TEXT,
            role        TEXT,
            location    TEXT,
            salary      TEXT,
            skills      TEXT,
            raw_jd      TEXT,
            match_notes TEXT,
            created_at  TEXT
        )
    """)
    con.commit(); con.close()

# FIX #3: return rowid from the SAME connection that did the insert
def save_job(company, role, location, salary, skills, raw_jd, match_notes) -> int:
    con = _con()
    cur = con.execute(
        "INSERT INTO jobs (company,role,location,salary,skills,raw_jd,match_notes,created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (company, role, location, salary, json.dumps(skills),
         raw_jd, match_notes, datetime.utcnow().isoformat())
    )
    jid = cur.lastrowid
    con.commit(); con.close()
    return jid

def all_jobs():
    con = _con()
    rows = con.execute(
        "SELECT id,company,role,location,salary,skills,created_at FROM jobs ORDER BY id DESC"
    ).fetchall()
    con.close(); return rows

# FIX #10: use row_factory so columns are addressable by name
def job_by_id(jid: int):
    con = _con()
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    con.close()
    return dict(row) if row else None

def delete_job(jid: int):
    con = _con()
    con.execute("DELETE FROM jobs WHERE id=?", (jid,))
    con.commit(); con.close()

# ── Resume ────────────────────────────────────────────────────────────────────
def load_resume() -> str | None:
    if os.path.exists("resume.txt"):
        with open("resume.txt", encoding="utf-8") as f:
            return f.read()
    return None

def save_resume(text: str):
    with open("resume.txt", "w", encoding="utf-8") as f:
        f.write(text)

# ── Gemini helpers ────────────────────────────────────────────────────────────
# FIX #9: exponential backoff retry on rate-limit / transient errors
def _ask_sync(system: str, user: str, retries: int = 3) -> str:
    delay = 2
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                )
            )
            return resp.text.strip()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Only retry on rate-limit or server errors
            if "429" in err_str or "quota" in err_str or "503" in err_str or "500" in err_str:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise  # non-retryable — surface immediately
    raise last_err

# FIX #2: run blocking Gemini call in a thread so the bot event loop stays free
async def _ask(system: str, user: str) -> str:
    return await asyncio.to_thread(_ask_sync, system, user)

# ── Prompts ───────────────────────────────────────────────────────────────────
EXTRACT_SYS = """You are a job description parser. Return ONLY valid JSON, no markdown fences, no explanation.

Schema:
{
  "company": "string or null",
  "role": "string",
  "location": "string or null",
  "salary": "string or null",
  "skills": ["technical skills only — languages, frameworks, tools, cloud, methodologies. No soft skills."],
  "summary": "2-sentence plain-English summary of what this role does"
}"""

async def extract_jd(jd_text: str) -> dict:
    raw = await _ask(EXTRACT_SYS, jd_text)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

MATCH_SYS = """You are a resume vs job description analyser. Return ONLY valid JSON, no markdown fences.

Schema:
{
  "strong": ["skills the candidate clearly has"],
  "missing": ["skills the candidate lacks"],
  "verdict": "one honest sentence about overall fit"
}"""

async def match_resume(resume_text: str, skills: list) -> dict:
    prompt = f"Resume:\n{resume_text}\n\nRequired skills: {', '.join(skills)}"
    raw = await _ask(MATCH_SYS, prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

PREP_SYS = """You are a technical interview coach. Return ONLY valid JSON, no markdown fences.

Schema:
{
  "topics": ["5 core technical topics to study"],
  "questions": ["5 likely interview questions"],
  "tip": "one sharp, specific piece of advice for this exact role"
}"""

async def generate_prep(jd_text: str, role: str) -> dict:
    prompt = f"Role: {role}\n\nJD:\n{jd_text}"
    raw = await _ask(PREP_SYS, prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── State: pending flows ──────────────────────────────────────────────────────
# FIX #4 + #6: unified pending state for both /add and /resume flows
pending: dict[int, str] = {}  # chat_id -> "add" | "resume"

# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # FIX #4: clear any stuck pending state on /start
    pending.pop(update.effective_chat.id, None)
    await update.message.reply_text(
        "👋 *Career Bot*\n\n"
        "`/resume` — set or view your resume\n"
        "`/add` — save a job description\n"
        "`/list` — all saved jobs\n"
        "`/prep <id>` — interview prep for a job\n"
        "`/stats` — skill frequency \\+ gaps\n"
        "`/delete <id>` — remove a job\n\n"
        "Start with `/resume` to paste your resume\\.",
        parse_mode="MarkdownV2"
    )

async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # FIX #6: if no inline args, wait for NEXT message as resume text
    if not ctx.args:
        resume = load_resume()
        if resume:
            preview = esc(resume[:500]) + ("…" if len(resume) > 500 else "")
            await update.message.reply_text(
                f"📄 *Current resume \\(preview\\):*\n\n{preview}\n\n"
                "Send `/resume` then paste new text to update\\.",
                parse_mode="MarkdownV2"
            )
        else:
            pending[update.effective_chat.id] = "resume"
            await update.message.reply_text(
                "📝 Paste your full resume text now \\(as a plain message\\):",
                parse_mode="MarkdownV2"
            )
        return

    # Inline: /resume <text> still works for short resumes
    save_resume(" ".join(ctx.args))
    await update.message.reply_text("✅ Resume saved\\!", parse_mode="MarkdownV2")

async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending[update.effective_chat.id] = "add"
    await update.message.reply_text("📋 Paste the job description now:")

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()

    state = pending.pop(chat_id, None)

    if state == "resume":
        if len(text) < 20:
            await update.message.reply_text("That looks too short for a resume. Try again with `/resume`.", parse_mode="Markdown")
            return
        save_resume(text)
        await update.message.reply_text("✅ Resume saved!")
        return

    if state == "add" or len(text) > 200:
        await process_jd(update, text)
        return

    await update.message.reply_text("Use /add to paste a JD, or /help for commands.")

async def process_jd(update: Update, jd_text: str):
    msg = await update.message.reply_text("⏳ Analysing job description…")

    # Step 1: extract
    try:
        parsed = await extract_jd(jd_text)
    except json.JSONDecodeError as e:
        await msg.edit_text(f"❌ Gemini returned malformed JSON: {e}\nTry again.")
        return
    except Exception as e:
        await msg.edit_text(f"❌ AI call failed: {e}")
        return

    company  = (parsed.get("company") or "Unknown").strip()
    role     = (parsed.get("role") or "Unknown").strip()
    location = (parsed.get("location") or "—").strip()
    salary   = (parsed.get("salary") or "—").strip()
    skills   = parsed.get("skills") or []
    summary  = (parsed.get("summary") or "").strip()

    # Validate skills is actually a list of strings
    if not isinstance(skills, list):
        skills = []
    skills = [str(s).strip() for s in skills if s]

    # Step 2: resume match (non-fatal if it fails)
    match_notes = ""
    resume = load_resume()
    if resume and skills:
        try:
            m = await match_resume(resume, skills)
            strong  = ", ".join(m.get("strong", [])) or "—"
            missing = ", ".join(m.get("missing", [])) or "—"
            verdict = m.get("verdict", "")
            match_notes = f"Strong: {strong}\nMissing: {missing}\nVerdict: {verdict}"
        except Exception as e:
            match_notes = f"Match analysis failed: {e}"

    # Step 3: save
    jid = save_job(company, role, location, salary, skills, jd_text, match_notes)

    # Step 4: reply — FIX #5: escape all LLM output
    skills_str = esc(", ".join(skills)) if skills else "—"
    reply = (
        f"✅ *Saved as Job \\#{jid}*\n\n"
        f"🏢 {esc(company)} — {esc(role)}\n"
        f"📍 {esc(location)}  💰 {esc(salary)}\n\n"
        f"📝 {esc(summary)}\n\n"
        f"🛠 *Skills:* {skills_str}\n"
    )
    if match_notes:
        reply += f"\n📊 *Resume Match:*\n{esc(match_notes)}\n"
    reply += f"\nRun `/prep {jid}` for interview prep\\."

    await msg.edit_text(reply, parse_mode="MarkdownV2")

async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    jobs = all_jobs()
    if not jobs:
        await update.message.reply_text("No jobs saved yet\\. Use /add\\.", parse_mode="MarkdownV2")
        return
    lines = ["📋 *Saved Jobs*\n"]
    for row in jobs:
        jid, company, role, location, salary, _, created_at = row
        lines.append(f"*\\#{jid}* {esc(company)} \\| {esc(role)} \\| {esc(location)} \\| {created_at[:10]}")
    await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")

async def cmd_prep(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: `/prep <id>`", parse_mode="MarkdownV2")
        return
    try:
        jid = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("Provide a numeric job ID\\.", parse_mode="MarkdownV2")
        return

    row = job_by_id(jid)
    if not row:
        await update.message.reply_text(f"No job \\#{jid} found\\.", parse_mode="MarkdownV2")
        return

    company = row["company"]
    role    = row["role"]
    raw_jd  = row["raw_jd"]

    msg = await update.message.reply_text(f"⏳ Generating prep for {role} @ {company}…")

    try:
        prep = await generate_prep(raw_jd, role)
    except json.JSONDecodeError:
        await msg.edit_text("❌ AI returned malformed JSON\\. Try again\\.", parse_mode="MarkdownV2")
        return
    except Exception as e:
        await msg.edit_text(f"❌ Failed: {esc(str(e))}", parse_mode="MarkdownV2")
        return

    topics    = "\n".join(f"  • {esc(t)}" for t in prep.get("topics", []))
    questions = "\n".join(f"  {i+1}\\. {esc(q)}" for i, q in enumerate(prep.get("questions", [])))
    tip       = esc(prep.get("tip", ""))

    reply = (
        f"🎯 *Interview Prep — {esc(role)} @ {esc(company)}*\n\n"
        f"📚 *Study Topics:*\n{topics}\n\n"
        f"❓ *Likely Questions:*\n{questions}\n\n"
        f"💡 *Tip:* {tip}"
    )
    await msg.edit_text(reply, parse_mode="MarkdownV2")

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    jobs = all_jobs()
    if not jobs:
        await update.message.reply_text("No jobs saved yet\\.", parse_mode="MarkdownV2")
        return

    all_skills = []
    for row in jobs:
        _, _, _, _, _, skills_json, _ = row
        try:
            all_skills.extend(json.loads(skills_json))
        except (json.JSONDecodeError, TypeError):  # FIX #8: specific exception
            pass

    if not all_skills:
        await update.message.reply_text("No skills extracted yet\\.", parse_mode="MarkdownV2")
        return

    counts = Counter(all_skills).most_common(15)
    lines = [f"📊 *Skill Frequency \\({len(jobs)} jobs saved\\)*\n"]
    for skill, count in counts:
        bar = "█" * min(count, 10)
        lines.append(f"`{skill:<22}` {bar} {count}")

    resume = load_resume()
    if resume:
        resume_lower = resume.lower()
        missing = [s for s, _ in counts if s.lower() not in resume_lower]
        if missing:
            lines.append(f"\n🔴 *You're missing \\(most requested\\):*")
            lines.append(esc(", ".join(missing[:8])))

    await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")

async def cmd_delete(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: `/delete <id>`", parse_mode="MarkdownV2")
        return
    try:
        jid = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("Provide a numeric ID\\.", parse_mode="MarkdownV2")
        return
    delete_job(jid)
    await update.message.reply_text(f"🗑 Job \\#{jid} deleted\\.", parse_mode="MarkdownV2")

# ── FIX #1: main() is NOT async — run_polling() is sync and manages its own loop
def main():
    init_db()
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_start))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("add",    cmd_add))
    app.add_handler(CommandHandler("list",   cmd_list))
    app.add_handler(CommandHandler("prep",   cmd_prep))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("delete", cmd_delete))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot running…")
    app.run_polling()   # sync — manages its own event loop correctly

if __name__ == "__main__":
    main()