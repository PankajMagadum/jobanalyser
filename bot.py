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
print("=== ENV CHECK ===")
_REQUIRED = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "GOOGLE_SHEET_ID", "GOOGLE_SERVICE_JSON"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    print(f"FATAL: missing env vars: {_missing}")
    raise SystemExit(1)
print("All env vars present. Starting…")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]
SHEET_ID       = os.environ["GOOGLE_SHEET_ID"]
GSERVICE_JSON  = os.environ["GOOGLE_SERVICE_JSON"]

gemini_client = genai.Client(api_key=GEMINI_KEY)
MODEL = "gemini-2.5-flash"

# ── Google Sheets ─────────────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Canonical column order — never change without also fixing _save_job_sync
HEADERS = [
    "ID", "Company", "Role", "Location", "Salary",
    "Skills", "Apply Link", "Strong", "Missing",
    "Verdict", "Summary", "Date Added", "Raw JD"
]

def _gc() -> gspread.Client:
    info = json.loads(GSERVICE_JSON)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def _jobs_sheet() -> gspread.Worksheet:
    sh = _gc().open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("Jobs")
        # Verify header matches — if not, this is a stale sheet from an old version
        existing = ws.row_values(1)
        if existing != HEADERS:
            # Rename old sheet, create fresh one
            ws.update_title("Jobs_old")
            ws = sh.add_worksheet(title="Jobs", rows=2000, cols=len(HEADERS))
            ws.append_row(HEADERS, value_input_option="RAW")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Jobs", rows=2000, cols=len(HEADERS))
        ws.append_row(HEADERS, value_input_option="RAW")
    return ws

def _resume_sheet() -> gspread.Worksheet:
    sh = _gc().open_by_key(SHEET_ID)
    try:
        return sh.worksheet("Resume")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Resume", rows=10, cols=2)
        ws.append_row(["Key", "Value"])
        return ws

# ── DB ops ────────────────────────────────────────────────────────────────────
def _next_id(ws) -> int:
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
                   apply_link, strong, missing, verdict, summary, raw_jd) -> int:
    ws = _jobs_sheet()
    jid = _next_id(ws)
    ws.append_row([
        jid,
        company,
        role,
        location,
        salary,
        ", ".join(skills) if skills else "",
        apply_link,                                    # empty string if skipped
        ", ".join(strong) if strong else "",
        ", ".join(missing) if missing else "",
        verdict,
        summary,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        raw_jd[:500],                                  # first 500 chars for reference
    ], value_input_option="RAW")
    return jid

def _all_jobs_sync() -> list[dict]:
    return _jobs_sheet().get_all_records()

def _job_by_id_sync(jid: int) -> dict | None:
    for row in _jobs_sheet().get_all_records():
        try:
            if int(row.get("ID", -1)) == jid:
                return row
        except (ValueError, TypeError):
            pass
    return None

def _delete_job_sync(jid: int) -> bool:
    ws = _jobs_sheet()
    for i, row in enumerate(ws.get_all_values()):
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
    ws = _resume_sheet()
    for i, row in enumerate(ws.get_all_values()):
        if row and row[0] == "resume":
            ws.update_cell(i + 1, 2, text)
            return
    ws.append_row(["resume", text])

def _load_resume_sync() -> str | None:
    for row in _resume_sheet().get_all_values():
        if row and row[0] == "resume":
            return row[1] if len(row) > 1 else None
    return None

# ── Async wrappers ────────────────────────────────────────────────────────────
async def db_save_job(company, role, location, salary, skills,
                      apply_link, strong, missing, verdict, summary, raw_jd) -> int:
    return await asyncio.to_thread(
        _save_job_sync, company, role, location, salary,
        skills, apply_link, strong, missing, verdict, summary, raw_jd
    )

async def db_all_jobs() -> list[dict]:
    return await asyncio.to_thread(_all_jobs_sync)

async def db_job_by_id(jid: int) -> dict | None:
    return await asyncio.to_thread(_job_by_id_sync, jid)

async def db_delete_job(jid: int) -> bool:
    return await asyncio.to_thread(_delete_job_sync, jid)

async def db_save_resume(text: str):
    await asyncio.to_thread(_save_resume_sync, text)

async def db_load_resume() -> str | None:
    return await asyncio.to_thread(_load_resume_sync)

# ── Gemini ────────────────────────────────────────────────────────────────────
# What each call does:
#   extract_jd    → parses raw JD text into structured fields (company/role/skills/summary etc.)
#   match_resume  → compares your resume against the skill list → strong / missing / verdict
#   generate_prep → produces 5 study topics + 5 interview questions + one tip for the role

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
            if any(c in str(e).lower() for c in ["429", "quota", "503", "500"]):
                if attempt < retries - 1:
                    time.sleep(delay); delay *= 2; continue
            raise
    raise last_err

async def _ask(system: str, user: str) -> str:
    return await asyncio.to_thread(_ask_sync, system, user)

def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    return json.loads(raw.strip())

EXTRACT_SYS = """You are a job description parser.
Return ONLY a valid JSON object — no markdown fences, no explanation, nothing else.

Schema:
{
  "company": "company name or null",
  "role": "job title",
  "location": "city/country or null",
  "salary": "salary info or null",
  "skills": ["Python", "AWS", "Docker"],
  "summary": "Exactly 2 sentences: what this role does day-to-day."
}

Rules:
- skills: technical only (languages, frameworks, tools, cloud, databases). No soft skills.
- Null for missing fields. No extra keys."""

async def extract_jd(jd_text: str) -> dict:
    return _parse_json(await _ask(EXTRACT_SYS, jd_text))

MATCH_SYS = """You are a resume analyser.
Return ONLY a valid JSON object — no markdown fences, no explanation.

Schema:
{
  "strong": ["skills present in the resume"],
  "missing": ["required skills absent from resume"],
  "verdict": "One honest sentence about overall fit."
}"""

async def match_resume_ai(resume_text: str, skills: list) -> dict:
    prompt = f"Resume:\n{resume_text}\n\nRequired skills: {', '.join(skills)}"
    return _parse_json(await _ask(MATCH_SYS, prompt))

PREP_SYS = """You are a technical interview coach.
Return ONLY a valid JSON object — no markdown fences, no explanation.

Schema:
{
  "topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "questions": ["q1", "q2", "q3", "q4", "q5"],
  "tip": "One sharp, specific tip for this exact role."
}"""

async def generate_prep(role: str, skills: str, summary: str) -> dict:
    prompt = f"Role: {role}\nRequired skills: {skills}\nRole summary: {summary}"
    return _parse_json(await _ask(PREP_SYS, prompt))

# ── Flow state ────────────────────────────────────────────────────────────────
# Stored in memory — survives within a deployment, resets on restart (acceptable)
# Keys: chat_id → {"step": "jd"|"link"|"resume", "jd_text": str, "parsed": dict, "match": dict}
pending: dict[int, dict] = {}

# ── Telegram handlers ─────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending.pop(update.effective_chat.id, None)
    await update.message.reply_text(
        "👋 Career Bot\n\n"
        "/resume — set or view your resume\n"
        "/add — save a job description\n"
        "/list — all saved jobs\n"
        "/prep <id> — interview prep for a job\n"
        "/stats — skill frequency + gaps\n"
        "/delete <id> — remove a job\n\n"
        "Start with /resume to paste your resume."
    )

async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        resume = await db_load_resume()
        if resume:
            preview = resume[:500] + ("…" if len(resume) > 500 else "")
            await update.message.reply_text(
                f"📄 Current resume (preview):\n\n{preview}\n\n"
                "Send /resume again then paste new text to update."
            )
        else:
            pending[update.effective_chat.id] = {"step": "resume"}
            await update.message.reply_text("📝 Paste your full resume text now:")
        return
    await db_save_resume(" ".join(ctx.args))
    await update.message.reply_text("✅ Resume saved!")

async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending[update.effective_chat.id] = {"step": "jd"}
    await update.message.reply_text(
        "📋 Paste the job description text now.\n"
        "Tip: copy only the role description, not the application form questions."
    )

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    state = pending.get(chat_id, {})
    step = state.get("step")

    # ── resume ────────────────────────────────────────────────────────────────
    if step == "resume":
        pending.pop(chat_id, None)
        if len(text) < 20:
            await update.message.reply_text("Too short. Try /resume again.")
            return
        msg = await update.message.reply_text("⏳ Saving resume…")
        await db_save_resume(text)
        await msg.edit_text("✅ Resume saved to Google Sheets!")
        return

    # ── step 1: receive JD text → analyse immediately → then ask for link ────
    if step == "jd" or (step is None and len(text) > 200):
        msg = await update.message.reply_text("⏳ Step 1/2 — Analysing job description…")

        try:
            parsed = await extract_jd(text)
        except Exception as e:
            await msg.edit_text(f"❌ Could not parse JD: {e}\nTry again with /add.")
            pending.pop(chat_id, None)
            return

        company  = str(parsed.get("company") or "Unknown").strip()
        role     = str(parsed.get("role") or "Unknown").strip()
        location = str(parsed.get("location") or "-").strip()
        salary   = str(parsed.get("salary") or "-").strip()
        skills   = [str(s).strip() for s in (parsed.get("skills") or []) if s]
        summary  = str(parsed.get("summary") or "").strip()

        # Resume match
        strong, missing, verdict = [], [], ""
        resume = await db_load_resume()
        if resume and skills:
            try:
                m = await match_resume_ai(resume, skills)
                strong  = m.get("strong") or []
                missing = m.get("missing") or []
                verdict = str(m.get("verdict") or "")
            except Exception as e:
                verdict = f"Match failed: {e}"
        elif not resume:
            verdict = "No resume set — use /resume to add one."

        # Store everything in pending — ask for link next
        pending[chat_id] = {
            "step": "link",
            "jd_text": text,
            "company": company, "role": role, "location": location,
            "salary": salary, "skills": skills, "summary": summary,
            "strong": strong, "missing": missing, "verdict": verdict,
        }

        skills_str  = ", ".join(skills) if skills else "-"
        strong_str  = ", ".join(strong) if strong else "-"
        missing_str = ", ".join(missing) if missing else "-"

        await msg.edit_text(
            f"✅ Parsed! Here's what I found:\n\n"
            f"🏢 {company} — {role}\n"
            f"📍 {location}  💰 {salary}\n\n"
            f"🛠 Skills: {skills_str}\n\n"
            f"📊 Resume Match:\n"
            f"  ✅ Strong: {strong_str}\n"
            f"  ❌ Missing: {missing_str}\n"
            f"  💬 {verdict}\n\n"
            f"📝 {summary}\n\n"
            f"🔗 Now send the apply link (or send: skip)"
        )
        return

    # ── step 2: receive apply link → save to sheet ────────────────────────────
    if step == "link":
        apply_link = "" if text.lower() == "skip" else text
        s = state  # has all parsed data from step 1

        msg = await update.message.reply_text("⏳ Step 2/2 — Saving to Google Sheets…")
        try:
            jid = await db_save_job(
                s["company"], s["role"], s["location"], s["salary"],
                s["skills"], apply_link,
                s["strong"], s["missing"], s["verdict"],
                s["summary"], s["jd_text"]
            )
        except Exception as e:
            await msg.edit_text(f"❌ Sheet write failed: {e}")
            return
        finally:
            pending.pop(chat_id, None)

        link_line = f"🔗 {apply_link}" if apply_link else "🔗 No link saved"
        await msg.edit_text(
            f"✅ Saved as Job #{jid}\n"
            f"🏢 {s['company']} — {s['role']}\n"
            f"{link_line}\n\n"
            f"Run /prep {jid} for interview prep.\n"
            f"Run /list to see all saved jobs."
        )
        return

    await update.message.reply_text("Use /add to save a job, or /help for all commands.")

async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Loading jobs…")
    try:
        jobs = await db_all_jobs()
    except Exception as e:
        await msg.edit_text(f"❌ Could not load jobs: {e}")
        return
    if not jobs:
        await msg.edit_text("No jobs saved yet. Use /add.")
        return
    lines = [f"📋 Saved Jobs ({len(jobs)} total)\n"]
    for row in jobs:
        jid     = row.get("ID", "?")
        company = row.get("Company", "?")
        role    = row.get("Role", "?")
        date    = str(row.get("Date Added", ""))
        link    = row.get("Apply Link", "")
        line    = f"#{jid}  {company} | {role} | {date}"
        if link:
            line += f"\n      Apply: {link}"
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
    try:
        row = await db_job_by_id(jid)
    except Exception as e:
        await msg.edit_text(f"❌ Could not load job: {e}")
        return
    if not row:
        await msg.edit_text(f"No job #{jid} found.")
        return

    role    = row.get("Role", "")
    company = row.get("Company", "")
    skills  = row.get("Skills", "")
    summary = row.get("Summary", "")

    await msg.edit_text(f"⏳ Generating interview prep for {role} @ {company}…")
    try:
        prep = await generate_prep(role, skills, summary)
    except Exception as e:
        await msg.edit_text(f"❌ Failed: {e}")
        return

    topics    = "\n".join(f"  • {t}" for t in prep.get("topics", []))
    questions = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(prep.get("questions", [])))
    tip       = prep.get("tip", "")

    await msg.edit_text(
        f"🎯 Interview Prep — {role} @ {company}\n\n"
        f"📚 Study Topics:\n{topics}\n\n"
        f"❓ Likely Questions:\n{questions}\n\n"
        f"💡 Tip: {tip}"
    )

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Crunching stats…")
    try:
        jobs = await db_all_jobs()
    except Exception as e:
        await msg.edit_text(f"❌ Could not load jobs: {e}")
        return
    if not jobs:
        await msg.edit_text("No jobs saved yet.")
        return

    all_skills = []
    for row in jobs:
        raw = str(row.get("Skills", ""))
        all_skills.extend([s.strip() for s in raw.split(",") if s.strip()])

    if not all_skills:
        await msg.edit_text("No skills found in saved jobs.")
        return

    counts = Counter(all_skills).most_common(15)
    lines = [f"📊 Skill Frequency ({len(jobs)} jobs)\n"]
    for skill, count in counts:
        bar = "█" * min(count, 10)
        lines.append(f"{skill:<25} {bar} {count}")

    resume = await db_load_resume()
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
    try:
        found = await db_delete_job(jid)
    except Exception as e:
        await msg.edit_text(f"❌ Delete failed: {e}")
        return
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
