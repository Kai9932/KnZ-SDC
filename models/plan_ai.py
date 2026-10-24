"""
models/plan_ai.py
Turns a career's missing skills into an ordered learning plan.

- Only skill NAMES, career title and preferences are sent to DeepSeek
  (never the resume).
- The AI reply is validated and normalised. If the API key is missing,
  the call fails or the JSON is bad, a deterministic fallback plan is
  returned so the page always works.
- No URLs are produced here. Resource links come in step 2.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import date

import httpx

log = logging.getLogger("plan_ai")

MAX_SKILLS = 12
LEVEL_HOURS = {"Beginner": 6, "Intermediate": 4, "Advanced": 3}
PHASE_NAMES = ["Foundations", "Core skills", "Applied practice", "Job readiness"]

SOFT_SKILLS = {
    "active listening", "speaking", "critical thinking", "complex problem solving",
    "coordination", "social perceptiveness", "persuasion", "negotiation",
    "service orientation", "instructing", "time management", "writing",
    "reading comprehension", "judgment and decision making", "learning strategies",
    "active learning", "management of personnel resources",
}

# Small hint table used only by the fallback plan.
TOPIC_HINTS = {
    "programming": "Programming fundamentals in Python",
    "data base user interface and query software": "SQL basics: SELECT, WHERE, JOIN",
    "mathematics": "Applied maths and statistics for work",
    "systems analysis": "Requirements gathering and system design basics",
    "systems evaluation": "Evaluating systems with metrics",
    "operations analysis": "Process analysis and improvement",
    "technology design": "Designing user-focused technology solutions",
    "troubleshooting": "Debugging and systematic troubleshooting",
    "quality control analysis": "Testing and quality assurance basics",
    "science": "Scientific method and data interpretation",
    "monitoring": "Monitoring and reporting with dashboards",
    "spreadsheet software": "Excel: formulas, pivot tables, charts",
    "analytical or scientific software": "Data analysis with Python (pandas)",
    "development environment software": "Git, IDEs and version control",
}

SYSTEM_PROMPT = """You are a curriculum designer for a career-reskilling app.
Reply with ONE valid json object and nothing else.

Rules:
- "skills" are O*NET skill names the learner is missing. Translate each into a concrete,
  learnable topic (e.g. "Data base user interface and query software" -> "SQL basics: SELECT, WHERE, JOIN").
- Use only skills from the list. Do not invent skills. Do not include URLs.
- Put prerequisites first. Create 3 to 4 phases.
- Each module: 1 to 10 hours, 3 to 4 short learning objectives.
- If weeks_available is given, total hours should not exceed weeks_available * hours_per_week.
- kind = "technical" for software/technical skills, "soft" for interpersonal or cognitive skills.
- search_query = a short phrase a learner would type to find a tutorial.
- Match difficulty to the learner level.

json format:
{
  "summary": "one or two sentences",
  "phases": [
    {"name": "Foundations", "focus": "short phrase",
     "modules": [
       {"title": "...", "skill": "<one of the skills>", "topic": "...", "kind": "technical",
        "level": "Beginner", "hours": 5, "objectives": ["...", "..."], "search_query": "..."}
     ]}
  ],
  "gaps": [{"skill": "<one of the skills>", "topic": "...", "priority": 90}]
}"""


# ---------- helpers ----------

def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _weeks_until(iso: str | None):
    if not iso:
        return None
    try:
        days = (date.fromisoformat(iso) - date.today()).days
    except ValueError:
        return None
    return max(1, math.ceil(days / 7)) if days > 0 else None


def _ask_deepseek(career, skills, prefs, weeks_available) -> dict:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    user_payload = {
        "target_career": career,
        "skills": skills,
        "level": prefs["level"],
        "hours_per_week": prefs["hours_per_week"],
        "learning_style": prefs["style"],
        "free_resources_only": prefs["free_only"],
        "weeks_available": weeks_available,
    }
    body = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "max_tokens": 3500,
    }
    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions"

    last_err = None
    for _ in range(2):  # one retry (JSON mode can occasionally return empty content)
        try:
            r = httpx.post(url, json=body, timeout=60,
                           headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"DeepSeek call failed: {last_err}")


def _fallback_raw(skills, level) -> dict:
    hours = LEVEL_HOURS.get(level, 5)
    size = max(1, math.ceil(len(skills) / 4))
    chunks = [skills[i:i + size] for i in range(0, len(skills), size)]
    phases, gaps = [], []
    for ci, chunk in enumerate(chunks):
        mods = []
        for s in chunk:
            topic = TOPIC_HINTS.get(s.lower(), s)
            soft = s.lower() in SOFT_SKILLS
            mods.append({
                "title": topic if topic != s else f"{s}: essentials",
                "skill": s, "topic": topic,
                "kind": "soft" if soft else "technical",
                "level": level, "hours": hours,
                "objectives": [
                    f"Understand the core ideas of {topic}",
                    "Complete one practical exercise",
                    "Explain how this skill is used in the target role",
                ],
                "search_query": f"{topic} tutorial for {level.lower()}s",
            })
            gaps.append({"skill": s, "topic": topic,
                         "priority": int(100 - len(gaps) * (60 / max(1, len(skills))))})
        phases.append({"name": PHASE_NAMES[min(ci, 3)], "focus": "", "modules": mods})
    return {"summary": "A standard plan built from your skill gaps, ordered by importance.",
            "phases": phases, "gaps": gaps}


def _normalise(raw: dict, skills, hpw: int, weeks_avail) -> dict:
    lookup = {s.lower(): s for s in skills}
    phases, counter = [], 0

    for ph in (raw.get("phases") or [])[:4]:
        mods = []
        for m in (ph.get("modules") or [])[:8]:
            title = str(m.get("title") or "").strip()
            if not title:
                continue
            counter += 1
            raw_skill = str(m.get("skill") or "").strip()
            skill = lookup.get(raw_skill.lower(), raw_skill or title)[:80]
            topic = str(m.get("topic") or title).strip()[:120]
            objectives = [str(o).strip()[:160] for o in (m.get("objectives") or []) if str(o).strip()][:5]
            if not objectives:
                objectives = [f"Understand the basics of {topic}"]
            kind = "soft" if (m.get("kind") == "soft" or skill.lower() in SOFT_SKILLS) else "technical"
            lvl = m.get("level") if m.get("level") in LEVEL_HOURS else "Beginner"
            mods.append({
                "id": f"m{counter}",
                "title": title[:120],
                "skill": skill,
                "topic": topic,
                "kind": kind,
                "level": lvl,
                "hours": round(_num(m.get("hours"), 1, 10, 4), 1),
                "objectives": objectives,
                "search_query": str(m.get("search_query") or topic).strip()[:120],
            })
        if mods:
            phases.append({"name": str(ph.get("name") or f"Phase {len(phases)+1}")[:40],
                           "focus": str(ph.get("focus") or "")[:80],
                           "modules": mods})
    if not phases:
        raise ValueError("No usable modules in plan")

    start = 1
    for ph in phases:
        hrs = sum(m["hours"] for m in ph["modules"])
        ph["hours"] = round(hrs, 1)
        ph["weeks"] = max(1, math.ceil(hrs / hpw))
        ph["start_week"] = start
        start += ph["weeks"]

    total_hours = round(sum(p["hours"] for p in phases), 1)
    total_weeks = start - 1

    gaps, seen = [], set()
    for g in raw.get("gaps") or []:
        s = lookup.get(str(g.get("skill") or "").strip().lower())
        if s and s not in seen:
            seen.add(s)
            gaps.append({"skill": s, "topic": str(g.get("topic") or s)[:120],
                         "priority": int(_num(g.get("priority"), 1, 100, 50))})
    for ph in phases:  # make sure every planned skill appears
        for m in ph["modules"]:
            if m["skill"] not in seen:
                seen.add(m["skill"])
                gaps.append({"skill": m["skill"], "topic": m["topic"],
                             "priority": max(10, 90 - len(gaps) * 10)})
    gaps.sort(key=lambda g: g["priority"], reverse=True)

    return {
        "summary": str(raw.get("summary") or "")[:300],
        "phases": phases,
        "gaps": gaps,
        "total_hours": total_hours,
        "total_weeks": total_weeks,
        "hours_per_week": hpw,
        "weeks_available": weeks_avail,
        "fits_deadline": None if weeks_avail is None else total_weeks <= weeks_avail,
    }


# ---------- public API ----------

def generate_plan(career: str, skills: list[str], level: str = "Beginner",
                  hours_per_week: int = 5, style: str = "Mixed",
                  free_only: bool = True, target_date: str | None = None) -> dict:
    skills = [s.strip() for s in skills if isinstance(s, str) and s.strip()][:MAX_SKILLS]
    if not skills:
        raise ValueError("No skills to plan for.")

    hpw = int(_num(hours_per_week, 1, 15, 5))
    weeks_avail = _weeks_until(target_date)
    prefs = {"level": level if level in LEVEL_HOURS else "Beginner",
             "hours_per_week": hpw, "style": style, "free_only": bool(free_only)}

    try:
        plan = _normalise(_ask_deepseek(career, skills, prefs, weeks_avail), skills, hpw, weeks_avail)
        plan["ai_generated"], plan["note"] = True, None
    except Exception as e:  # noqa: BLE001
        log.warning("AI plan failed, using fallback: %s", e)
        plan = _normalise(_fallback_raw(skills, prefs["level"]), skills, hpw, weeks_avail)
        plan["ai_generated"] = False
        plan["note"] = "AI is unavailable right now, so this is a standard plan."

    plan["career"] = career
    return plan