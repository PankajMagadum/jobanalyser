import os
import json
import asyncio
import time
import re
from datetime import datetime
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

# ── Config ──────────────────────────────────────────────────────────────────
print("=== ENV VARS AVAILABLE ===")
for k in sorted(os.environ.keys()):
    print(f"  {k}")
print("==========================")

_REQUIRED = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "GOOGLE_SHEET_ID", "GOOGLE_SERVICE_JSON"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    print(f"FATAL: missing env vars: {_missing}")
    raise SystemExit(1)

TELEGRAM_TOKEN  = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_KEY      = os.environ["GEMINI_API_KEY"]
SHEET_ID        = os.environ["GOOGLE_SHEET_ID"]        # the spreadsheet ID from the URL
GSERVICE_JSON   = os.environ["GOOGLE_SERVICE_JSON"]    # full service account JSON as a string

gemini_client = genai.Client(api_key=GEMINI_KEY)
MODEL = "gemini-2.5-flash"

# ── Google Sheets client ──────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
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
        ws = sh.add_worksheet(title="Jobs", rows=1000, cols=12)
        # Write header row
        ws.append_row([
            "ID", "Company", "Role", "Location", "Salary",
            "Skills", "Apply Link", "Strong", "Missing",
            "Verdict", "Summary", "Date Added"
        ], value_input_option="RAW")
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

# Column indices (1-based for gspread)
COL = {
    "id":         1,
    "company":    2,
    "role":       3,
    "location":   4,
    "salary":     5,
    "skills":     6,
    "apply_link": 7,
    "strong":     8,
    "missing":    9,
    "verdict":    10,
    "summary":    11,
    "date":       12,
}

# ── Sheet DB operations (all blocking — called via asyncio.to_thread) ─────────
def _next_id_sync() -> int:
    ws = _get_sheet()
    rows = ws.get_all_values()
    if len(rows) <= 1:   # only header
        return 1
    # Find max existing ID
    ids = []
    for row in rows[1:]:
        try: ids.append(int(row[0]))
        except (ValueError, IndexError): pass
    return max(ids) + 1 if ids else 1

def _save_job_sync(company, role, location, salary, skills,
                   apply_link, strong, missing, verdict, summary) -> int:
    ws = _get_sheet()
    jid = _next_id_sync()
    ws.append_row([
        jid,
        company,
        role,
        location,
        salary,
        ", ".join(skills),
        apply_link,
        ", ".join(strong),
        ", ".join(missing),
        verdict,
        summary,
        datetime.utcnow().strftime("%Y-%m-%d"),
    ], value_input_option="RAW")
    return jid

def _all_jobs_sync() -> list[dict]:
    ws = _get_sheet()
    rows = ws.get_all_records()  # returns list of dicts using header row as keys
    return rows

def _job_by_id_sync(jid: int) -> dict | None:
    ws = _get_sheet()
    rows = ws.get_all_records()
    for row in rows:
        if int(row.get("ID", -1)) == jid:
            return row
    return None

def _delete_job_sync(jid: int):
    ws = _get_sheet()
    rows = ws.get_all_values()  # includes header
    for i, row in enumerate(rows):
        if i == 0: continue  # skip header
        try:
            if int(row[0]) == jid:
                ws.delete_rows(i + 1)  # gspread rows are 1-indexed
                return True
        except (ValueError, IndexError):
            pass
    return False

def _save_resume_sync(text: str):
    ws = _get_resume_sheet()
    rows = ws.get_all_values()
    # Find existing resume row
    for i, row in enumerate(rows):
        if row and row[0] == "resume":
            ws.update_cell(i + 1, 2, text)
            return
    ws.append_row(["resume", text])

def _load_resume_sync() -> str | None:
    ws = _get_resume_sheet()
    rows = ws.get_all_values()
    for row in rows:
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

async def delete_job(jid: int):
    return await asyncio.to_thread(_delete_job_sync, jid)

async def save_resume(text: str):
    await asyncio.to_thread(_save_resume_sync, text)

async def load_resume() -> str | None:
    return await asyncio.to_thread(_load_resume_sync)

# ── Markdown escaping ─────────────────────────────────────────────────────────
_MD_SPECIAL = re.compile(r'([*_`\[\]()~>#+=|{}.!\\-])')

def esc(text: str) -> str:
    return _MD_SPECIAL.sub(r'\\\1', str(text)) if text else ""

# ── Gemini helpers ────────────────────────────────────────────────────────────
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
            if "429" in err_str or "quota" in err_str or "503" in err_str or "500" in err_str:
                if attempt < retries - 1:
                    time.sleep(delay); delay *= 2; continue
            raise
    raise last_err

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

async def match_resume_ai(resume_text: str, skills: list) -> dict:
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

# ── Multi-step flow state ─────────────────────────────────────────────────────
# pending[chat_id] = {"step": "jd"|"link"|"resume", "jd_text": str}
pending: dict[int, dict] = {}

# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
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
    if not ctx.args:
        resume = await load_resume()
        if resume:
            preview = esc(resume[:500] + ("…" if len(resume) > 500 else ""))
            await update.message.reply_text(
                f"📄 *Current resume \\(preview\\):*\n\n{preview}\n\n"
                "Send `/resume` again then paste new text to update\\.",
                parse_mode="MarkdownV2"
            )
        else:
            pending[update.effective_chat.id] = {"step": "resume"}
            await update.message.reply_text(
                "📝 Paste your full resume text now \\(as a plain message\\):",
                parse_mode="MarkdownV2"
            )
        return
    await save_resume(" ".join(ctx.args))
    await update.message.reply_text("✅ Resume saved\\!", parse_mode="MarkdownV2")

async def cmd_add(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pending[update.effective_chat.id] = {"step": "jd"}
    await update.message.reply_text("📋 Paste the job description now:")

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()
    state = pending.get(chat_id, {})
    step = state.get("step")

    # ── resume flow ───────────────────────────────────────────────────────────
    if step == "resume":
        pending.pop(chat_id, None)
        if len(text) < 20:
            await update.message.reply_text(
                "That looks too short\\. Try `/resume` again\\.",
                parse_mode="MarkdownV2"
            )
            return
        msg = await update.message.reply_text("⏳ Saving resume…")
        await save_resume(text)
        await msg.edit_text("✅ Resume saved to Google Sheets\\!", parse_mode="MarkdownV2")
        return

    # ── jd flow: step 1 — received JD text ───────────────────────────────────
    if step == "jd":
        pending[chat_id] = {"step": "link", "jd_text": text}
        await update.message.reply_text(
            "🔗 Now paste the *apply link* for this job\\.\n"
            "Or send `skip` to save without a link\\.",
            parse_mode="MarkdownV2"
        )
        return

    # ── jd flow: step 2 — received apply link ────────────────────────────────
    if step == "link":
        jd_text = state.get("jd_text", "")
        pending.pop(chat_id, None)
        apply_link = "" if text.lower() == "skip" else text
        await process_jd(update, jd_text, apply_link)
        return

    # ── fallback: long message treated as JD ─────────────────────────────────
    if len(text) > 200:
        pending[chat_id] = {"step": "link", "jd_text": text}
        await update.message.reply_text(
            "🔗 Got it\\. Now paste the *apply link* for this job\\.\n"
            "Or send `skip` to save without a link\\.",
            parse_mode="MarkdownV2"
        )
        return

    await update.message.reply_text("Use /add to paste a JD, or /help for commands\\.", parse_mode="MarkdownV2")

async def process_jd(update: Update, jd_text: str, apply_link: str):
    msg = await update.message.reply_text("⏳ Analysing job description…")

    # Extract
    try:
        parsed = await extract_jd(jd_text)
    except json.JSONDecodeError as e:
        await msg.edit_text(f"❌ Gemini returned bad JSON: {esc(str(e))}", parse_mode="MarkdownV2")
        return
    except Exception as e:
        await msg.edit_text(f"❌ AI call failed: {esc(str(e))}", parse_mode="MarkdownV2")
        return

    company  = (parsed.get("company") or "Unknown").strip()
    role     = (parsed.get("role") or "Unknown").strip()
    location = (parsed.get("location") or "—").strip()
    salary   = (parsed.get("salary") or "—").strip()
    skills   = parsed.get("skills") or []
    summary  = (parsed.get("summary") or "").strip()

    if not isinstance(skills, list):
        skills = []
    skills = [str(s).strip() for s in skills if s]

    # Resume match
    strong, missing, verdict = [], [], ""
    resume = await load_resume()
    if resume and skills:
        try:
            m = await match_resume_ai(resume, skills)
            strong  = m.get("strong", [])
            missing = m.get("missing", [])
            verdict = m.get("verdict", "")
        except Exception as e:
            verdict = f"Match failed: {e}"

    # Save to sheet
    await msg.edit_text("⏳ Saving to Google Sheets…")
    try:
        jid = await save_job(
            company, role, location, salary, skills,
            apply_link, strong, missing, verdict, summary
        )
    except Exception as e:
        await msg.edit_text(f"❌ Sheet write failed: {esc(str(e))}", parse_mode="MarkdownV2")
        return

    # Reply
    skills_str  = esc(", ".join(skills)) if skills else "—"
    strong_str  = esc(", ".join(strong)) if strong else "—"
    missing_str = esc(", ".join(missing)) if missing else "—"
    link_str    = f"[Apply here]({apply_link})" if apply_link else "—"

    reply = (
        f"✅ *Saved as Job \\#{jid}*\n\n"
        f"🏢 {esc(company)} — {esc(role)}\n"
        f"📍 {esc(location)}  💰 {esc(salary)}\n"
        f"🔗 {link_str}\n\n"
        f"📝 {esc(summary)}\n\n"
        f"🛠 *Skills:* {skills_str}\n\n"
        f"📊 *Resume Match:*\n"
        f"  ✅ Strong: {strong_str}\n"
        f"  ❌ Missing: {missing_str}\n"
        f"  💬 {esc(verdict)}\n\n"
        f"Run `/prep {jid}` for interview prep\\."
    )
    await msg.edit_text(reply, parse_mode="MarkdownV2")

async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Loading jobs…")
    jobs = await all_jobs()
    if not jobs:
        await msg.edit_text("No jobs saved yet\\. Use /add\\.", parse_mode="MarkdownV2")
        return
    lines = ["📋 *Saved Jobs*\n"]
    for row in jobs:
        jid      = row.get("ID", "?")
        company  = esc(row.get("Company", "?"))
        role     = esc(row.get("Role", "?"))
        location = esc(row.get("Location", "—"))
        date     = esc(str(row.get("Date Added", "")))
        link     = row.get("Apply Link", "")
        link_part = f" — [Apply]({link})" if link else ""
        lines.append(f"*\\#{jid}* {company} \\| {role} \\| {location} \\| {date}{link_part}")
    await msg.edit_text("\n".join(lines), parse_mode="MarkdownV2")

async def cmd_prep(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: `/prep <id>`", parse_mode="MarkdownV2")
        return
    try:
        jid = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("Provide a numeric job ID\\.", parse_mode="MarkdownV2")
        return

    msg = await update.message.reply_text("⏳ Loading job…")
    row = await job_by_id(jid)
    if not row:
        await msg.edit_text(f"No job \\#{jid} found\\.", parse_mode="MarkdownV2")
        return

    company = row.get("Company", "")
    role    = row.get("Role", "")
    # Reconstruct a JD-like text from what we have stored
    raw_jd = (
        f"Company: {company}\nRole: {role}\n"
        f"Location: {row.get('Location','')}\n"
        f"Skills required: {row.get('Skills','')}\n"
        f"Summary: {row.get('Summary','')}"
    )

    await msg.edit_text(f"⏳ Generating prep for {esc(role)} @ {esc(company)}…", parse_mode="MarkdownV2")

    try:
        prep = await generate_prep(raw_jd, role)
    except json.JSONDecodeError:
        await msg.edit_text("❌ AI returned bad JSON\\. Try again\\.", parse_mode="MarkdownV2")
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
    msg = await update.message.reply_text("⏳ Crunching stats…")
    jobs = await all_jobs()
    if not jobs:
        await msg.edit_text("No jobs saved yet\\.", parse_mode="MarkdownV2")
        return

    all_skills = []
    for row in jobs:
        raw = row.get("Skills", "")
        if raw:
            all_skills.extend([s.strip() for s in raw.split(",") if s.strip()])

    if not all_skills:
        await msg.edit_text("No skills extracted yet\\.", parse_mode="MarkdownV2")
        return

    counts = Counter(all_skills).most_common(15)
    lines = [f"📊 *Skill Frequency \\({len(jobs)} jobs saved\\)*\n"]
    for skill, count in counts:
        bar = "█" * min(count, 10)
        lines.append(f"`{skill:<22}` {bar} {esc(str(count))}")

    resume = await load_resume()
    if resume:
        resume_lower = resume.lower()
        missing = [s for s, _ in counts if s.lower() not in resume_lower]
        if missing:
            lines.append("\n🔴 *You're missing \\(most requested\\):*")
            lines.append(esc(", ".join(missing[:8])))

    await msg.edit_text("\n".join(lines), parse_mode="MarkdownV2")

async def cmd_delete(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text("Usage: `/delete <id>`", parse_mode="MarkdownV2")
        return
    try:
        jid = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("Provide a numeric ID\\.", parse_mode="MarkdownV2")
        return
    msg = await update.message.reply_text("⏳ Deleting…")
    found = await delete_job(jid)
    if found:
        await msg.edit_text(f"🗑 Job \\#{jid} deleted\\.", parse_mode="MarkdownV2")
    else:
        await msg.edit_text(f"Job \\#{jid} not found\\.", parse_mode="MarkdownV2")

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