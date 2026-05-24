import os
import json
import asyncio
import time
import re
from datetime import datetime, timezone
from collections import Counter

import gspread
from google.oauth2.service_account import Credentials
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters,
)
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
print("=== ENV VARS AVAILABLE ===")
for k in sorted(os.environ.keys()):
    print(f"  {k}")
print("==========================")

_REQUIRED = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "GOOGLE_SHEET_ID", "GOOGLE_SERVICE_JSON"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    print(f"FATAL: missing env vars: {_missing}")
    raise SystemExit(1)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]
SHEET_ID       = os.environ["GOOGLE_SHEET_ID"]
GSERVICE_JSON  = os.environ["GOOGLE_SERVICE_JSON"]

gemini_client = genai.Client(api_key=GEMINI_KEY)
MODEL = "gemini-2.5-flash"

# ── Markdown escaping ─────────────────────────────────────────────────────────
# MarkdownV2 requires escaping: _ * [ ] ( ) ~ ` > # + - = | { } . !
# We also escape the em-dash — since some outputs contain it
_MD_RE = re.compile(r'([\\_*\[\]()~`>#+\-=|{}.!])')

def esc(text: str) -> str:
    """Escape any string for safe use inside a MarkdownV2 message."""
    if not text:
        return ""
    # Replace em-dash with plain hyphen first (em-dash itself is fine but
    # Gemini sometimes outputs sequences that confuse the parser)
    text = str(text).replace("\u2014", "-").replace("\u2013", "-")
    return _MD_RE.sub(r'\\\1', text)

def safe_reply(text: str) -> str:
    """Build a plain-text fallback with no parse_mode, for error messages."""
    return text

# ── Google Sheets ─────────────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Sheet columns — keep in sync with header row
HEADERS = [
    "ID", "Company", "Role", "Location", "Salary",
    "Skills", "Apply Link", "Strong", "Missing",
    "Verdict", "Summary", "Date Added"
]

def _sheets_client() -> gspread.Client:
    info = json.loads(GSERVICE_JSON)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def _get_sheet() -> gspread.Worksheet:
    gc = _sheets_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("Jobs")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Jobs", rows=1000, cols=len(HEADERS))
        ws.append_row(HEADERS, value_input_option="RAW")
    return ws

def _get_resume_sheet() -> gspread.Worksheet:
    gc = _sheets_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("Resume")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Resume", rows=10, cols=2)
        ws.append_row(["Key", "Value"])
    return ws

# ── Sheet DB ops (all sync — called via asyncio.to_thread) ───────────────────
def _next_id_sync() -> int:
    ws = _get_sheet()
    rows = ws.get_all_values()
    if len(rows) <= 1:
        return 1
    ids = []
    for row in rows[1:]:
        try:
            ids.append(int(row[0]))
        except (ValueError, IndexError):
            pass
    return max(ids) + 1 if ids else 1

def _save_job_sync(company, role, location, salary, skills,
                   apply_link, strong, missing, verdict, summary) -> int:
    ws = _get_sheet()
    jid = _next_id_sync()
    row = [
        jid,
        company,
        role,
        location,
        salary,
        ", ".join(skills) if skills else "",
        apply_link,
        ", ".join(strong) if strong else "",
        ", ".join(missing) if missing else "",
        verdict,
        summary,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    ]
    ws.append_row(row, value_input_option="RAW")
    return jid

def _all_jobs_sync() -> list[dict]:
    ws = _get_sheet()
    return ws.get_all_records()

def _job_by_id_sync(jid: int) -> dict | None:
    ws = _get_sheet()
    for row in ws.get_all_records():
        try:
            if int(row.get("ID", -1)) == jid:
                return row
        except (ValueError, TypeError):
            pass
    return None

def _delete_job_sync(jid: int) -> bool:
    ws = _get_sheet()
    rows = ws.get_all_values()
    for i, row in enumerate(rows):
        if i == 0:
            continue
        try:
            if int(row[0]) == jid:
                ws.delete_rows(i + 1)
                return True
        except (ValueError, IndexError):
            pass
    return False

def _save_resume_sync(text: str):
    ws = _get_resume_sheet()
    rows = ws.get_all_values()
    for i, row in enumerate(rows):
        if row and row[0] == "resume":
            ws.update_cell(i + 1, 2, text)
            return
    ws.append_row(["resume", text])

def _load_resume_sync() -> str | None:
    ws = _get_resume_sheet()
    for row in ws.get_all_values():
        if row and row[0] == "resume":
            return row[1] if len(row) > 1 else None
    return None

# ── Async wrappers ────────────────────────────────────────────────────────────
async def save_job(company, role, location, salary, skills,
                   apply_link, strong, missing, verdict, summary) -> int:
    return await asyncio.to_thread(
        _save_job_sync, company, role, location, salary, skills,
        apply_link, strong, missing, verdict, summary
    )

async def all_jobs() -> list[dict]:
    return await asyncio.to_thread(_all_jobs_sync)

async def job_by_id(jid: int) -> dict | None:
    return await asyncio.to_thread(_job_by_id_sync, jid)

async def delete_job(jid: int) -> bool:
    return await asyncio.to_thread(_delete_job_sync, jid)

async def save_resume(text: str):
    await asyncio.to_thread(_save_resume_sync, text)

async def load_resume() -> str | None:
    return await asyncio.to_thread(_load_resume_sync)

# ── Gemini ────────────────────────────────────────────────────────────────────
# What Gemini does in this bot:
#   1. extract_jd   — reads raw JD text, returns structured JSON
#                     (company, role, location, salary, skills list, 2-sentence summary)
#   2. match_resume — compares your resume text against the skill list,
#                     returns what you have (strong), what you lack (missing), one-line verdict
#   3. generate_prep — given the JD, produces 5 study topics, 5 interview questions, one tip

def _ask_sync(system: str, user: str, retries: int = 3) -> str:
    delay = 2
    last_err = None
    for attempt in range(retries):
        try:
            resp = gemini_client.models.generate_content(
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
            if any(code in err_str for code in ["429", "quota", "503", "500"]):
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise
    raise last_err

async def _ask(system: str, user: str) -> str:
    return await asyncio.to_thread(_ask_sync, system, user)

def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    return json.loads(raw.strip())

EXTRACT_SYS = """You are a job description parser. 
Return ONLY valid JSON — no markdown fences, no explanation, nothing else.

Schema:
{
  "company": "company name string, or null if not found",
  "role": "job title string",
  "location": "city/country or null",
  "salary": "salary range string or null",
  "skills": ["Python", "AWS", "Docker"],
  "summary": "Exactly 2 sentences describing what this role does day-to-day."
}

Rules:
- skills: technical only (languages, frameworks, tools, cloud, databases, methodologies). No soft skills.
- If a field is not present in the JD, use null.
- Do not include any text outside the JSON object."""

async def extract_jd(jd_text: str) -> dict:
    raw = await _ask(EXTRACT_SYS, jd_text)
    return _parse_json(raw)

MATCH_SYS = """You are a resume analyser.
Return ONLY valid JSON — no markdown fences, no explanation.

Schema:
{
  "strong": ["skills clearly present in the resume"],
  "missing": ["skills required by the job but absent from resume"],
  "verdict": "One honest sentence about overall fit for this role."
}"""

async def match_resume_ai(resume_text: str, skills: list) -> dict:
    prompt = f"Resume:\n{resume_text}\n\nRequired skills: {', '.join(skills)}"
    raw = await _ask(MATCH_SYS, prompt)
    return _parse_json(raw)

PREP_SYS = """You are a technical interview coach.
Return ONLY valid JSON — no markdown fences, no explanation.

Schema:
{
  "topics": ["topic 1", "topic 2", "topic 3", "topic 4", "topic 5"],
  "questions": ["question 1", "question 2", "question 3", "question 4", "question 5"],
  "tip": "One sharp, specific piece of advice for this exact role."
}"""

async def generate_prep(jd_text: str, role: str) -> dict:
    prompt = f"Role: {role}\n\nJD:\n{jd_text}"
    raw = await _ask(PREP_SYS, prompt)
    return _parse_json(raw)

# ── State ─────────────────────────────────────────────────────────────────────
pending: dict[int, dict] = {}  # chat_id -> {step, jd_text?}

# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending.pop(update.effective_chat.id, None)
    text = (
        "👋 Career Bot\n\n"
        "/resume — set or view your resume\n"
        "/add — save a job description\n"
        "/list — all saved jobs\n"
        "/prep <id> — interview prep for a job\n"
        "/stats — skill frequency + gaps\n"
        "/delete <id> — remove a job\n\n"
        "Start with /resume to paste your resume."
    )
    await update.message.reply_text(text)

async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        resume = await load_resume()
        if resume:
            preview = resume[:500] + ("…" if len(resume) > 500 else "")
            await update.message.reply_text(
                f"📄 Current resume (preview):\n\n{preview}\n\n"
                "Send /resume again then paste new text to update."
            )
        else:
            pending[update.effective_chat.id] = {"step": "resume"}
            await update.message.reply_text(
                "📝 Paste your full resume text now (as a plain message):"
            )
        return
    await save_resume(" ".join(ctx.args))
    await update.message.reply_text("✅ Resume saved!")

async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending[update.effective_chat.id] = {"step": "jd"}
    await update.message.reply_text(
        "📋 Paste the job description text now.\n"
        "(Copy the job description text only — not the application form)"
    )

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    state = pending.get(chat_id, {})
    step = state.get("step")

    if step == "resume":
        pending.pop(chat_id, None)
        if len(text) < 20:
            await update.message.reply_text(
                "That looks too short. Try /resume again."
            )
            return
        msg = await update.message.reply_text("⏳ Saving resume…")
        await save_resume(text)
        await msg.edit_text("✅ Resume saved to Google Sheets!")
        return

    if step == "jd":
        pending[chat_id] = {"step": "link", "jd_text": text}
        await update.message.reply_text(
            "🔗 Now send the apply link for this job.\n"
            "Or send: skip"
        )
        return

    if step == "link":
        jd_text = state.get("jd_text", "")
        pending.pop(chat_id, None)
        apply_link = "" if text.lower() == "skip" else text
        await process_jd(update, jd_text, apply_link)
        return

    if len(text) > 200:
        pending[chat_id] = {"step": "link", "jd_text": text}
        await update.message.reply_text(
            "🔗 Got the JD. Now send the apply link.\nOr send: skip"
        )
        return

    await update.message.reply_text("Use /add to paste a JD, or /help for commands.")

async def process_jd(update: Update, jd_text: str, apply_link: str):
    msg = await update.message.reply_text("⏳ Step 1/3 — Extracting job details…")

    try:
        parsed = await extract_jd(jd_text)
    except json.JSONDecodeError as e:
        await msg.edit_text(f"❌ Gemini returned bad JSON: {e}\nTry again.")
        return
    except Exception as e:
        await msg.edit_text(f"❌ AI call failed: {e}")
        return

    company  = str(parsed.get("company") or "Unknown").strip()
    role     = str(parsed.get("role") or "Unknown").strip()
    location = str(parsed.get("location") or "-").strip()
    salary   = str(parsed.get("salary") or "-").strip()
    skills   = parsed.get("skills") or []
    summary  = str(parsed.get("summary") or "").strip()

    if not isinstance(skills, list):
        skills = []
    skills = [str(s).strip() for s in skills if s]

    await msg.edit_text("⏳ Step 2/3 — Matching against your resume…")

    strong, missing, verdict = [], [], ""
    resume = await load_resume()
    if resume and skills:
        try:
            m = await match_resume_ai(resume, skills)
            strong  = m.get("strong") or []
            missing = m.get("missing") or []
            verdict = str(m.get("verdict") or "")
        except Exception as e:
            verdict = f"Match failed: {e}"
    elif not resume:
        verdict = "No resume set. Use /resume to add one."

    await msg.edit_text("⏳ Step 3/3 — Saving to Google Sheets…")

    try:
        jid = await save_job(
            company, role, location, salary, skills,
            apply_link, strong, missing, verdict, summary
        )
    except Exception as e:
        await msg.edit_text(f"❌ Sheet write failed: {e}")
        return

    # Build reply in plain text — no MarkdownV2 complexity
    skills_str  = ", ".join(skills) if skills else "-"
    strong_str  = ", ".join(strong) if strong else "-"
    missing_str = ", ".join(missing) if missing else "-"
    link_line   = f"🔗 {apply_link}" if apply_link else "🔗 No link saved"

    reply = (
        f"✅ Saved as Job #{jid}\n\n"
        f"🏢 {company} — {role}\n"
        f"📍 {location}  💰 {salary}\n"
        f"{link_line}\n\n"
        f"📝 {summary}\n\n"
        f"🛠 Skills: {skills_str}\n\n"
        f"📊 Resume Match:\n"
        f"  ✅ Strong: {strong_str}\n"
        f"  ❌ Missing: {missing_str}\n"
        f"  💬 {verdict}\n\n"
        f"Run /prep {jid} for interview prep."
    )
    await msg.edit_text(reply)

async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Loading jobs…")
    jobs = await all_jobs()
    if not jobs:
        await msg.edit_text("No jobs saved yet. Use /add.")
        return
    lines = ["📋 Saved Jobs\n"]
    for row in jobs:
        jid     = row.get("ID", "?")
        company = row.get("Company", "?")
        role    = row.get("Role", "?")
        date    = str(row.get("Date Added", ""))
        link    = row.get("Apply Link", "")
        line    = f"#{jid} {company} | {role} | {date}"
        if link:
            line += f"\n    🔗 {link}"
        lines.append(line)
    await msg.edit_text("\n".join(lines))

async def cmd_prep(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /prep <id>")
        return
    try:
        jid = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("Provide a numeric job ID.")
        return

    msg = await update.message.reply_text("⏳ Loading job…")
    row = await job_by_id(jid)
    if not row:
        await msg.edit_text(f"No job #{jid} found.")
        return

    company = row.get("Company", "")
    role    = row.get("Role", "")
    raw_jd  = (
        f"Company: {company}\nRole: {role}\n"
        f"Location: {row.get('Location','')}\n"
        f"Skills required: {row.get('Skills','')}\n"
        f"Summary: {row.get('Summary','')}"
    )

    await msg.edit_text(f"⏳ Generating prep for {role} @ {company}…")

    try:
        prep = await generate_prep(raw_jd, role)
    except json.JSONDecodeError:
        await msg.edit_text("❌ AI returned bad JSON. Try again.")
        return
    except Exception as e:
        await msg.edit_text(f"❌ Failed: {e}")
        return

    topics    = "\n".join(f"  • {t}" for t in prep.get("topics", []))
    questions = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(prep.get("questions", [])))
    tip       = prep.get("tip", "")

    reply = (
        f"🎯 Interview Prep — {role} @ {company}\n\n"
        f"📚 Study Topics:\n{topics}\n\n"
        f"❓ Likely Questions:\n{questions}\n\n"
        f"💡 Tip: {tip}"
    )
    await msg.edit_text(reply)

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Crunching stats…")
    jobs = await all_jobs()
    if not jobs:
        await msg.edit_text("No jobs saved yet.")
        return

    all_skills = []
    for row in jobs:
        raw = row.get("Skills", "")
        if raw:
            all_skills.extend([s.strip() for s in str(raw).split(",") if s.strip()])

    if not all_skills:
        await msg.edit_text("No skills extracted yet.")
        return

    counts = Counter(all_skills).most_common(15)
    lines = [f"📊 Skill Frequency ({len(jobs)} jobs saved)\n"]
    for skill, count in counts:
        bar = "█" * min(count, 10)
        lines.append(f"{skill:<25} {bar} {count}")

    resume = await load_resume()
    if resume:
        resume_lower = resume.lower()
        missing = [s for s, _ in counts if s.lower() not in resume_lower]
        if missing:
            lines.append("\n🔴 You're missing (most requested):")
            lines.append(", ".join(missing[:8]))

    await msg.edit_text("\n".join(lines))

async def cmd_delete(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: /delete <id>")
        return
    try:
        jid = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("Provide a numeric ID.")
        return
    msg = await update.message.reply_text("⏳ Deleting…")
    found = await delete_job(jid)
    await msg.edit_text(f"🗑 Job #{jid} deleted." if found else f"Job #{jid} not found.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
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
    app.run_polling()

if __name__ == "__main__":
    main()