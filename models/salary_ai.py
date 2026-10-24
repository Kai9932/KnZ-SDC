import json
import logging
import os
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger("salary_ai")

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "salary_cache.json"
MODEL = "deepseek-chat"


def _client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=40)


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        tmp.replace(CACHE_PATH)
    except OSError:
        logger.exception("Could not write salary cache")


def _valid(item: dict) -> bool:
    try:
        lo, med, hi = (int(item["min"]), int(item["median"]), int(item["max"]))
        return 0 < lo <= med <= hi < 1_000_000 and item.get("confidence") in {"low", "medium", "high"}
    except (KeyError, TypeError, ValueError):
        return False


def _ask_deepseek(client, missing: list[dict]) -> dict:
    listing = "\n".join(f'- {c["O*NET-SOC Code"]}: {c["Title"]}' for c in missing)

    system = (
        "You are a Malaysian labour-market analyst. Reply with a single JSON object only."
    )
    user = f"""Estimate typical gross MONTHLY salaries in Malaysian Ringgit (RM) for these
occupations as they exist in Malaysia (private and public sector, all experience levels).

{listing}

Return exactly this JSON shape:
{{"salaries": [
  {{"code": "<code as given>", "min": <int>, "median": <int>, "max": <int>,
    "confidence": "low|medium|high",
    "note": "<max 15 words, e.g. how common the role is in Malaysia>"}}
]}}

Rules:
- min = typical entry-level, median = typical mid-career, max = typical senior level.
- Whole numbers, RM per month, no currency symbols.
- If the occupation is rare or barely exists in Malaysia, still give your best
  estimate but set confidence to "low" and say so in the note.
- Do not invent precision: round to the nearest RM 100."""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1500,
    )
    data = json.loads(resp.choices[0].message.content)
    out = {}
    for item in data.get("salaries", []):
        if _valid(item):
            out[str(item["code"])] = {
                "min": int(item["min"]),
                "median": int(item["median"]),
                "max": int(item["max"]),
                "confidence": item["confidence"],
                "note": str(item.get("note", ""))[:120],
            }
    return out


def apply_ai_salaries(careers: list[dict]) -> list[dict]:
    """Adds c['ai_salary'] (dict or None) to each career. Never raises."""
    cache = _load_cache()
    missing = [c for c in careers if c["O*NET-SOC Code"] not in cache]

    if missing:
        client = _client()
        if client is None:
            logger.warning("DEEPSEEK_API_KEY not set; using fallback wages")
        else:
            try:
                fresh = _ask_deepseek(client, missing)
                if fresh:
                    cache.update(fresh)
                    _save_cache(cache)
            except Exception:
                logger.exception("DeepSeek salary lookup failed")

    for c in careers:
        c["ai_salary"] = cache.get(c["O*NET-SOC Code"])
    return careers