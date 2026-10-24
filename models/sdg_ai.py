"""SDG impact analysis for one career + the user's skills.

Flow: try DeepSeek (JSON mode) -> validate everything it returns -> if the key is
missing, the call fails, or the output is unusable, fall back to keyword rules.
Nothing numeric (savings, emissions) is ever generated: only goal/target links
and short explanations.
"""
import json
import os
import re

import requests

GOALS = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health & Well-being",
    4: "Quality Education", 5: "Gender Equality", 6: "Clean Water & Sanitation",
    7: "Affordable & Clean Energy", 8: "Decent Work & Economic Growth",
    9: "Industry, Innovation & Infrastructure", 10: "Reduced Inequalities",
    11: "Sustainable Cities & Communities", 12: "Responsible Consumption & Production",
    13: "Climate Action", 14: "Life Below Water", 15: "Life on Land",
    16: "Peace, Justice & Strong Institutions", 17: "Partnerships for the Goals",
}

# Goal-level explanations used by the fallback (and as the default target).
GOAL_REASON = {
    3: "supports health services and better care decisions",
    4: "supports learning, training and access to skills",
    7: "helps organisations use energy more efficiently",
    8: "supports productive work and better business decisions",
    9: "drives innovation, technology and infrastructure",
    11: "supports planning for safer, better-run communities",
    12: "helps reduce waste and use resources responsibly",
    13: "helps track, plan for and reduce environmental impact",
}
DEFAULT_TARGET = {3: "3.8", 4: "4.4", 7: "7.3", 8: "8.2", 9: "9.5", 11: "11.3", 12: "12.6", 13: "13.3"}

KEYWORDS = {
    3: ["health", "nurs", "medical", "clinic", "patient", "care", "pharma"],
    4: ["teach", "train", "educat", "learn", "instruct", "curriculum", "tutor"],
    7: ["energy", "power", "solar", "grid", "electric", "utility"],
    8: ["business", "manage", "analyst", "finance", "account", "sales", "operations", "hr", "market", "data"],
    9: ["software", "engineer", "develop", "program", "python", "sql", "network", "system",
        "machine learning", "ai", "database", "automation", "technolog"],
    11: ["urban", "construct", "transport", "architect", "civil", "planning"],
    12: ["supply", "procure", "manufactur", "logistic", "sustainab", "quality", "waste"],
    13: ["environment", "climate", "carbon", "emission", "sustainab", "forecast", "predict"],
}

_TARGET_RE = re.compile(r"^(\d{1,2})\.\d{1,2}[a-h]?$")


def _score_goals(text: str) -> list[int]:
    text = text.lower()
    scores = {g: sum(1 for k in kws if k in text) for g, kws in KEYWORDS.items()}
    ranked = [g for g, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0]
    return ranked or [8, 9, 4]


def _fallback(title: str, have: list[str], missing: list[str]) -> dict:
    skills_all = [(s, "have") for s in have] + [(s, "missing") for s in missing]
    top = _score_goals(" ".join([title] + have + missing))[:4]
    sdgs = [{"id": g, "name": GOALS[g], "target": DEFAULT_TARGET.get(g),
             "reason": f"Work as a {title} {GOAL_REASON.get(g, 'contributes to this goal')}."} for g in top]
    skills = []
    for name, status in skills_all[:8]:
        ids = _score_goals(name)[:2]
        ids = [i for i in ids if i in top] or top[:1]
        skills.append({"name": name, "status": status, "sdgs": ids,
                       "reason": f"{name} {GOAL_REASON.get(ids[0], 'supports this goal')}."})
    return {
        "sdgs": sdgs,
        "contribution": (f"As a {title}, you can apply your skills to help organisations make "
                         f"better decisions for people, the economy and the environment."),
        "skills": skills,
        "source": "rules",
    }


def _ask_deepseek(title: str, have: list[str], missing: list[str]) -> dict:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    goals_list = "; ".join(f"{i}={n}" for i, n in GOALS.items())
    system = (
        "You map careers and skills to the UN Sustainable Development Goals for a Malaysian "
        "career-guidance app. Reply with JSON only. Use only real SDG numbers and real UN target "
        "numbers (e.g. 8.2). Never invent statistics, savings or emission figures. Keep reasons "
        "factual, plain and under 25 words. If unsure of the exact target, set target to null."
    )
    user = json.dumps({
        "career": title,
        "skills_user_has": have[:12],
        "skills_user_needs": missing[:12],
        "goals": goals_list,
        "return_format": {
            "sdgs": "3 to 5 items: {id:int, target:str|null, reason:str}",
            "contribution": "2 sentences, second person, how this career can contribute",
            "skills": "one item per skill listed above: {name:str, sdgs:[int,...max 2], reason:str}",
        },
    })
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "deepseek-chat", "temperature": 0.2,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=45,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def _clean(raw: dict, title: str, have: list[str], missing: list[str]) -> dict:
    """Keep only what we can trust; raise ValueError if nothing usable."""
    sdgs, seen = [], set()
    for item in raw.get("sdgs", []):
        try:
            gid = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if gid not in GOALS or gid in seen:
            continue
        seen.add(gid)
        target = item.get("target")
        m = _TARGET_RE.match(str(target)) if target else None
        target = str(target) if m and int(m.group(1)) == gid else None
        sdgs.append({"id": gid, "name": GOALS[gid], "target": target,
                     "reason": str(item.get("reason", ""))[:220]})
    if len(sdgs) < 2:
        raise ValueError("too few valid SDGs")
    sdgs = sdgs[:5]

    status = {s.lower(): (s, "have") for s in have}
    status.update({s.lower(): (s, "missing") for s in missing})
    skills = []
    for item in raw.get("skills", []):
        key = str(item.get("name", "")).lower()
        if key not in status:          # ignore skills the model made up
            continue
        ids = [int(i) for i in item.get("sdgs", []) if str(i).isdigit() and int(i) in GOALS][:2]
        if not ids:
            continue
        name, st = status[key]
        skills.append({"name": name, "status": st, "sdgs": ids,
                       "reason": str(item.get("reason", ""))[:220]})

    return {"sdgs": sdgs, "contribution": str(raw.get("contribution", ""))[:500],
            "skills": skills, "source": "deepseek"}


def analyse(title: str, have: list[str], missing: list[str]) -> dict:
    try:
        return _clean(_ask_deepseek(title, have, missing), title, have, missing)
    except Exception:
        return _fallback(title, have, missing)
