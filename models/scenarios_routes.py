
import math
import requests
import numpy as np
from datetime import date

SC_DIR = DATA_DIR / "scenarios"
SC_DIR.mkdir(parents=True, exist_ok=True)
SC_EXPOSURE_FILE = SC_DIR / "exposure_cache.json"
SC_YEARS = 5
SC_SPEED = {"moderate": 1.0, "fast": 2.0} 
DEEPSEEK_MODEL = "deepseek-chat"


def _deepseek_json(prompt: str, timeout: int = 45):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": DEEPSEEK_MODEL, "temperature": 0.2,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:
        logger.exception("DeepSeek call failed")
        return None


def _sc_series():
    p = DATA_DIR / "wages_occupation.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Occupation wage dataset not found.")
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df = df[(df["sex"] == "both") & (df["occupation"] != "overall")]
    df = df.dropna(subset=["year", "wages_median"])
    df["year"] = df["year"].astype(int)
    latest_year = int(df["year"].max())
    series = {}
    for name, g in df.groupby("occupation"):
        by = dict(sorted(zip(g["year"], g["wages_median"].astype(float))))
        if latest_year in by and len(by) >= 5:
            series[name] = by
    return latest_year, series


def _fit(ys, vs, future):
    """Log-linear trend fitted to the data, with a real 90% prediction interval."""
    x = np.array(ys, float)
    y = np.log(np.array(vs, float))
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    s = float(np.sqrt(((y - (a + b * x)) ** 2).sum() / max(n - 2, 1)))
    sxx = ((x - x.mean()) ** 2).sum()
    out = []
    for f in future:
        se = s * math.sqrt(1 + 1 / n + (f - x.mean()) ** 2 / sxx)
        out.append((a + b * f, se))
    return float(b), out


def _backtest(by):
    """Hide the last 2 years, forecast them, report average % error."""
    ys = list(by)
    if len(ys) < 6:
        return None
    train = ys[:-2][-8:]
    _, pred = _fit(train, [by[y] for y in train], ys[-2:])
    errs = [abs(math.exp(m) - by[y]) / by[y] for (m, _), y in zip(pred, ys[-2:])]
    return round(float(np.mean(errs)) * 100, 1)


def _exposure(names):
    """AI exposure per occupation, estimated by DeepSeek once and cached on disk."""
    try:
        cache = json.loads(SC_EXPOSURE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    missing = [n for n in names if n not in cache]
    if missing:
        d = _deepseek_json(
            "You are a labour economist. For each Malaysian occupation group below, give rough 0-100 "
            "judgements for the next 5 years: 'automation' = share of its tasks AI could replace, "
            "'augmentation' = share where AI could raise worker productivity. Base this on widely known "
            "research on generative-AI task exposure. Do not quote statistics. 'why' = one plain sentence. "
            'Return JSON: {"items":[{"name":"<exact name>","automation":n,"augmentation":n,"why":"..."}]}. '
            "Occupation groups: " + json.dumps(missing))
        for it in (d or {}).get("items", []):
            try:
                if it["name"] in missing:
                    cache[it["name"]] = {
                        "automation": int(max(0, min(100, float(it["automation"])))),
                        "augmentation": int(max(0, min(100, float(it["augmentation"])))),
                        "why": str(it.get("why", ""))[:220],
                        "by": f"DeepSeek ({DEEPSEEK_MODEL})", "date": date.today().isoformat()}
            except (KeyError, ValueError, TypeError):
                continue
        SC_EXPOSURE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return {n: cache[n] for n in names if n in cache}


def _sc_build() -> dict:
    latest_year, series = _sc_series()
    fits = {}
    for name, by in series.items():
        ys = list(by)[-8:]
        b, _ = _fit(ys, [by[y] for y in ys], [])
        fits[name] = (ys, b)
    spread = max(float(np.std([b for _, b in fits.values()])), 0.005) 
    expo = _exposure(list(series))

    out = []
    for name, by in series.items():
        ys, b = fits[name]
        _, base = _fit(ys, [by[y] for y in ys], [latest_year + t for t in range(1, SC_YEARS + 1)])
        e = expo.get(name)
        net = (e["augmentation"] - e["automation"]) / 100 if e else 0.0
        latest = by[latest_year]
        scen = {}
        for key, speed in SC_SPEED.items():
            shift = net * speed * spread           
            pts = []
            for t, (m, se) in enumerate(base, 1):
                m2 = m + shift * t
                pts.append({"year": latest_year + t, "mid": round(math.exp(m2), 1),
                            "lo": round(math.exp(m2 - 1.645 * se), 1),
                            "hi": round(math.exp(m2 + 1.645 * se), 1)})
            scen[key] = {"change_pct": round((pts[-1]["mid"] / latest - 1) * 100, 1), "points": pts}
        out.append({
            "name": name, "latest": latest,
            "hist_growth_pct": round((math.exp(b) - 1) * 100, 2),
            "ai_adjustment_pp": round(net * SC_SPEED["moderate"] * spread * 100, 2),
            "exposure": e, "backtest_mape_pct": _backtest(by),
            "history": [{"year": int(y), "median": v} for y, v in by.items()],
            "scenarios": scen})
    return {
        "latest_year": latest_year,
        "source": "DOSM occupational wages (data/wages_occupation.csv) + AI exposure estimated by DeepSeek",
        "method": "Trend = log-linear fit to the last 8 years of median wages (a proxy for demand); the shaded "
                  "band is the 90% prediction interval from how well that fit matched the data. Each scenario "
                  "adds an AI effect = (augmentation - automation) x adoption speed x the real spread of "
                  "growth between occupations. Exposure scores are an LLM judgement, not an official statistic.",
        "occupations": out}


def _sc_user_career(request: Request):
    """The signed-in user's target career: latest saved plan item, else top quiz match."""
    quiz = _load_quiz_result(request)
    if not quiz:
        return None, []
    plan = _load_plan(request)
    if plan:
        return plan[-1]["title"], plan[-1].get("missing_skills") or []
    c = quiz["careers"][0]
    return next((str(v) for k, v in c.items() if "title" in k.lower()), None), []


@app.get("/scenarios", response_class=HTMLResponse)
async def scenarios_page(request: Request):
    return templates.TemplateResponse(request=request, name="scenarios.html",
                                      context={"request": request})


@app.get("/api/scenarios")
def scenarios_api():
    return JSONResponse(_sc_build(), headers={"Cache-Control": "no-store"})


@app.get("/api/scenarios/me")
def scenarios_me(request: Request):
    title, skills = _sc_user_career(request)
    if not title:
        return {"occupation": None}
    names = [o["name"] for o in _sc_build()["occupations"]]
    d = _deepseek_json(f'Pick the ONE occupation group from the list that best fits the career "{title}". '
                       'Return JSON {"occupation": "<exact item from the list>"}. List: ' + json.dumps(names))
    occ = (d or {}).get("occupation")
    return {"career": title, "occupation": occ if occ in names else None, "skills": skills}


@app.get("/api/scenarios/insight")
def scenarios_insight(request: Request, occupation: str, personal: bool = False):
    data = _sc_build()
    occ = next((o for o in data["occupations"] if o["name"] == occupation), None)
    if not occ:
        return JSONResponse({"error": "Occupation not found."}, status_code=404)
    m, f = occ["scenarios"]["moderate"], occ["scenarios"]["fast"]
    facts = {"occupation": occupation, "latest_year": data["latest_year"], "years": SC_YEARS,
             "moderate_change_pct": m["change_pct"], "fast_change_pct": f["change_pct"],
             "historical_growth_pct_per_year": occ["hist_growth_pct"],
             "ai_exposure": occ["exposure"], "backtest_error_pct": occ["backtest_mape_pct"]}
    if personal:
        title, skills = _sc_user_career(request)
        facts["user_target_career"], facts["user_skills_being_built"] = title, skills

    fallback = [
        f"Median pay for this group is estimated to change {m['change_pct']}% (moderate AI) or {f['change_pct']}% (fast AI) over {SC_YEARS} years.",
        f"Past wage growth was about {occ['hist_growth_pct']}% a year; the shaded band shows how uncertain the forecast is.",
        "These are estimates from past data and an AI judgement, not guarantees. Compare futures, then talk to a career adviser."]
    d = _deepseek_json(
        "Write exactly 3 short plain-English takeaways (max 28 words each) for a Malaysian worker reading this "
        "scenario forecast. If user_target_career is present, make the third one about how their skills relate "
        "to the scenarios. Use ONLY the facts given: do not invent jobs, courses, employers or statistics. "
        'Say these are estimates. Return JSON {"takeaways":["...","...","..."]}.\n' + json.dumps(facts))
    items = [str(x) for x in (d or {}).get("takeaways", [])][:3]
    if len(items) == 3:
        return {"takeaways": items, "by": f"DeepSeek ({DEEPSEEK_MODEL})"}
    return {"takeaways": fallback, "by": "template"}