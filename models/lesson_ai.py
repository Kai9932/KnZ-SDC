"""models/lesson_ai.py

Career-specific lessons with real learning links, plus a short quiz.

DeepSeek never writes URLs (models invent dead links). It only picks a
provider and a search query; this file builds the URL from fixed templates.
"""
import os
import re
import json
import logging
from urllib.parse import quote_plus

import requests

log = logging.getLogger("lesson_ai")
MODEL = "deepseek-chat"

# Skill -> direct W3Schools tutorial home (checked longest key first).
W3_DIRECT = {
    "python": "https://www.w3schools.com/python/",
    "pandas": "https://www.w3schools.com/python/pandas/",
    "numpy": "https://www.w3schools.com/python/numpy/",
    "sql": "https://www.w3schools.com/sql/",
    "html": "https://www.w3schools.com/html/",
    "css": "https://www.w3schools.com/css/",
    "javascript": "https://www.w3schools.com/js/",
    "typescript": "https://www.w3schools.com/typescript/",
    "java": "https://www.w3schools.com/java/",
    "c++": "https://www.w3schools.com/cpp/",
    "php": "https://www.w3schools.com/php/",
    "react": "https://www.w3schools.com/react/",
    "node.js": "https://www.w3schools.com/nodejs/",
    "git": "https://www.w3schools.com/git/",
    "excel": "https://www.w3schools.com/excel/",
    "mongodb": "https://www.w3schools.com/mongodb/",
    "bootstrap": "https://www.w3schools.com/bootstrap5/",
    "django": "https://www.w3schools.com/django/",
    "statistics": "https://www.w3schools.com/statistics/",
    "data science": "https://www.w3schools.com/datascience/",
    "cybersecurity": "https://www.w3schools.com/cybersecurity/",
    "artificial intelligence": "https://www.w3schools.com/ai/",
}

# provider -> (URL builder, is_free)
PROVIDERS = {
    "youtube": (lambda q: "https://www.youtube.com/results?search_query=" + quote_plus(q + " tutorial"), True),
    "freecodecamp": (lambda q: "https://www.freecodecamp.org/news/search/?query=" + quote_plus(q), True),
    "mdn": (lambda q: "https://developer.mozilla.org/en-US/search?q=" + quote_plus(q), True),
    "khan": (lambda q: "https://www.khanacademy.org/search?page_search_query=" + quote_plus(q), True),
    "w3schools": (lambda q: "https://www.google.com/search?q=" + quote_plus("site:w3schools.com " + q), True),
    "google": (lambda q: "https://www.google.com/search?q=" + quote_plus(q + " tutorial for beginners"), True),
    "coursera": (lambda q: "https://www.coursera.org/search?query=" + quote_plus(q), False),
    "edx": (lambda q: "https://www.edx.org/search?q=" + quote_plus(q), False),
}


def _ask(prompt: str, timeout: int = 60) -> dict:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is not set.")
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "temperature": 0.3,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        log.exception("DeepSeek call failed")
        raise ValueError("The AI tutor is unavailable right now. Please try again.") from e


def _direct_w3(skill: str):
    s = skill.lower()
    for k in sorted(W3_DIRECT, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9+])" + re.escape(k) + r"(?![a-z0-9+])", s):
            return W3_DIRECT[k]
    return None


def generate_lesson(career: str, skill: str, level: str, style: str,
                    hours_per_week: int, free_only: bool) -> dict:
    allowed = [p for p, (_, free) in PROVIDERS.items() if free or not free_only]
    d = _ask(
        f'You are a career tutor. The learner wants to become a "{career}" and must learn the skill "{skill}". '
        f"Level: {level}. Learning style: {style}. Study time: {hours_per_week} hours/week.\n"
        f"Write a lesson about \"{skill}\" as it is actually used in the job of {career}. "
        "Return JSON with exactly these keys:\n"
        '{"summary": "2-3 plain sentences: what it is and why a ' + career + ' needs it",'
        ' "steps": ["4-6 short study steps in order, each starts with a verb"],'
        ' "practice": "one small hands-on task relevant to the career",'
        ' "resources": [{"title": "short title", "provider": "one of ' + json.dumps(allowed) + '",'
        ' "query": "a precise search query (3-8 words) that finds a good tutorial", '
        '"type": "video|article|course|docs", "why": "max 12 words"}]}\n'
        "Give 5 resources, mixing at least one video (youtube) and one reading source. "
        "Do NOT include any URLs. Do not invent course names; use search queries.")

    resources = []
    direct = _direct_w3(skill)
    if direct:
        resources.append({"title": f"W3Schools {skill} tutorial", "provider": "w3schools",
                          "type": "article", "url": direct,
                          "why": "Step-by-step lessons with examples you can try."})
    for r in d.get("resources", []):
        prov = str(r.get("provider", "")).lower()
        query = str(r.get("query", "")).strip()[:120]
        if prov not in allowed or not query:
            continue
        resources.append({
            "title": str(r.get("title", query))[:90],
            "provider": prov,
            "type": r.get("type") if r.get("type") in ("video", "article", "course", "docs") else "article",
            "url": PROVIDERS[prov][0](query),
            "why": str(r.get("why", ""))[:100],
        })
    if not resources:
        resources.append({"title": f"Search: {skill} tutorial", "provider": "google", "type": "article",
                          "url": PROVIDERS["google"][0](f"{skill} {career}"), "why": "General search."})

    steps = [str(s)[:200] for s in d.get("steps", [])][:6]
    if not steps:
        raise ValueError("The AI returned an empty lesson. Please try again.")
    return {"skill": skill, "career": career, "summary": str(d.get("summary", ""))[:600],
            "steps": steps, "practice": str(d.get("practice", ""))[:300],
            "resources": resources[:6]}


def generate_quiz(career: str, skill: str, level: str, summary: str = "") -> list:
    d = _ask(
        f'Write a 5-question multiple-choice quiz that checks a {level} learner\'s understanding of "{skill}" '
        f'for someone becoming a "{career}". Lesson summary: {summary}\n'
        "Rules: 4 options per question, exactly one correct, plausible wrong options, test understanding "
        "not trivia, no trick questions. Return JSON: "
        '{"questions":[{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"one sentence"}]} '
        "where answer is the 0-based index of the correct option.")
    out = []
    for q in d.get("questions", []):
        try:
            opts = [str(o)[:160] for o in q["options"]]
            ans = int(q["answer"])
            if len(opts) == 4 and 0 <= ans < 4 and str(q["question"]).strip():
                out.append({"question": str(q["question"])[:300], "options": opts, "answer": ans,
                            "explanation": str(q.get("explanation", ""))[:240]})
        except (KeyError, ValueError, TypeError):
            continue
    if len(out) < 3:
        raise ValueError("Could not build a quiz. Please try again.")
    return out[:5]