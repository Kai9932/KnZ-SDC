import re
from functools import lru_cache

import numpy as np
import pandas as pd

from models.career_matcher import (
    RIASEC_COLUMNS, PROJECT_DIR, load_occupations, load_essential_skills,
    load_software_skills, load_malaysia_wages, match_wage_occupation,
    normalise_text, safe_number, flag_is_true,
)
from models.salary_ai import apply_ai_salaries

DATA_DIR = PROJECT_DIR / "data"
GENERIC = {"software", "or", "and", "of", "the", "tools", "tool", "systems", "system", "for"}


@lru_cache(maxsize=1)
def _occ(): return load_occupations()
@lru_cache(maxsize=1)
def _ess(): return load_essential_skills()
@lru_cache(maxsize=1)
def _soft(): return load_software_skills()
@lru_cache(maxsize=1)
def _wages(): return load_malaysia_wages()


def _clean(o):
    """Make the result JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    return o


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d == 0 or len(np.unique(a)) == 1:      # flat profile = no real preference
        return None
    return round(float(np.clip(a @ b / d, 0, 1)) * 100, 1)


# ---------- skills ----------
def _rows(df, code):
    if df is None:
        return pd.DataFrame(columns=["Element Name"])
    r = df[df["O*NET-SOC Code"] == code]
    return r[r["Element Name"] != ""]


def _found(name, raw, norm):
    """Keyword match: exact phrase, or an acronym such as CAD / GIS."""
    words = [w for w in normalise_text(name).split() if w not in GENERIC]
    phrase = " ".join(words)
    if phrase and f" {phrase} " in f" {norm} ":
        return True
    return any(re.search(rf"\b{a}\b", raw) for a in re.findall(r"\b[A-Z]{2,5}\b", name))


def _tags(r):
    t = []
    if "Hot Technology" in r and flag_is_true(r["Hot Technology"]): t.append("Hot technology")
    if "In Demand" in r and flag_is_true(r["In Demand"]): t.append("In demand")
    return t


def _top_names(code, k=8):
    e = _rows(_ess(), code)
    if "importance" in e.columns:
        e = e.sort_values("importance", ascending=False)
    return list(dict.fromkeys(e["Element Name"].head(k)))


def _skills(code, raw):
    norm = normalise_text(raw) if raw else ""
    out = []

    def status(name):
        if raw is None: return "unknown"
        return "found" if _found(name, raw, norm) else "missing"

    e, scale = _rows(_ess(), code), 100
    if "importance" in e.columns and not e.empty:
        e = e.sort_values("importance", ascending=False)
        scale = 5 if e["importance"].max() <= 5 else 100
    for _, r in e.head(8).iterrows():
        imp = safe_number(r.get("importance"))
        out.append({"name": r["Element Name"], "kind": "Core skill", "tags": [],
                    "importance": None if np.isnan(imp) else round(imp / scale * 100),
                    "status": status(r["Element Name"])})

    sw, hot = _rows(_soft(), code).copy(), []
    if not sw.empty:
        sw["tags"] = sw.apply(_tags, axis=1)
        sw["p"] = sw["tags"].str.len()
        sw = sw.sort_values(["p", "Element Name"], ascending=[False, True]).drop_duplicates("Element Name")
        hot = [n for n, t in zip(sw["Element Name"], sw["tags"]) if t]
        for _, r in sw.head(8).iterrows():
            out.append({"name": r["Element Name"], "kind": "Software", "tags": r["tags"],
                        "importance": None, "status": status(r["Element Name"])})
    return out, hot


# ---------- wages (OpenDOSM) ----------
def _csv(name):
    p = DATA_DIR / name
    if not p.exists(): return None
    try: df = pd.read_csv(p)
    except Exception: return None
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"date", "wages_median"} <= set(df.columns): return None
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    if "sex" in df.columns:
        df = df[df["sex"].astype(str).str.lower().isin({"both", "both sexes", "overall", "all"})]
    df["wages_median"] = pd.to_numeric(df["wages_median"], errors="coerce")
    return df.dropna(subset=["year", "wages_median"])


def _label(df, prefer):
    for c in prefer:
        if c in df.columns:
            return c
    for c in df.columns:                      # partial match, e.g. "industry_group"
        if any(p in c for p in prefer):
            return c
    skip = {"date", "sex", "year", "wages_median", "wages_mean"}
    return next((c for c in df.columns
                 if c not in skip and not pd.api.types.is_numeric_dtype(df[c])), None)


def _pts(g):
    s = g.groupby("year")["wages_median"].mean()
    return [{"x": int(y), "y": round(float(v))} for y, v in s.items()]


def _all(df, col):
    if df is None or col is None or col not in df.columns:
        return []
    return [{"name": str(n), "series": _pts(g)} for n, g in df.groupby(col)
            if normalise_text(n) not in {"overall", "total", "all"}]


def _salary(code, title, wage, occ, ind):
    res = {"group": None, "group_series": [], "industries": [], "ai": None}
    group = wage.get("wage_matched_occupation")
    if occ is not None and group:
        col = _label(occ, ["occupation"])
        if col:
            g = occ[occ[col].astype(str).map(normalise_text) == normalise_text(group)]
            res["group"], res["group_series"] = str(group), _pts(g)
    if ind is not None:
        res["industries"] = _all(ind, _label(ind, ["industry", "sector"]))
    try:
        res["ai"] = apply_ai_salaries([{"O*NET-SOC Code": code, "Title": title}])[0].get("ai_salary")
    except Exception:
        res["ai"] = None
    return res


def _outlook(occ, group, hot):
    out = {"growth_pct": None, "since": None, "rank": None, "groups": 0, "hot": hot[:10]}
    if occ is None or not group: return out
    ch = {}
    for g in _all(occ, _label(occ, ["occupation"])):
        s = {p["x"]: p["y"] for p in g["series"]}
        if not s: continue
        last = max(s); first = last - 3
        if s.get(first):
            ch[normalise_text(g["name"])] = ((s[last] - s[first]) / s[first] * 100, first)
    me = ch.get(normalise_text(group))
    if me:
        out.update(growth_pct=round(me[0], 1), since=me[1], groups=len(ch),
                   rank=sorted(ch.values(), key=lambda v: -v[0]).index(me) + 1)
    return out


# ---------- related roles ----------
def _related(code, ovec, uvec, n=4):
    occ = _occ()
    M = occ[RIASEC_COLUMNS].to_numpy(float)
    sims = M @ ovec / (np.linalg.norm(M, axis=1) * np.linalg.norm(ovec) + 1e-9)
    mine = _top_names(code)
    out = []
    for i in [i for i in np.argsort(-sims) if occ.iloc[i]["O*NET-SOC Code"] != code][:n]:
        r = occ.iloc[i]
        theirs = _top_names(r["O*NET-SOC Code"])
        shared = [s for s in theirs if s in mine]
        w = match_wage_occupation(_wages(), r["Title"])
        out.append({
            "code": r["O*NET-SOC Code"], "title": r["Title"],
            "description": str(r["Description"])[:200],
            "match": _cos(uvec, M[i]),
            "overlap": shared, "extra": [s for s in theirs if s not in mine],
            "wage": w.get("malaysia_median_wage"),
            "why": f"{len(shared)} of {len(theirs)} core skills overlap with this career, "
                   f"and the interest profiles are {sims[i] * 100:.0f}% similar.",
        })
    return out


# ---------- main entry ----------
def build_detail(code, user_scores, resume_text):
    occ = _occ()
    m = occ[occ["O*NET-SOC Code"] == code]
    if m.empty:
        raise KeyError(code)
    row = m.iloc[0]
    uvec = np.array([user_scores[c] for c in RIASEC_COLUMNS], float)
    ovec = row[RIASEC_COLUMNS].to_numpy(float)

    skills, hot = _skills(code, resume_text)
    wage = match_wage_occupation(_wages(), row["Title"])
    occ_df, ind_df = _csv("wages_occupation.csv"), _csv("wages_industry.csv")
    salary = _salary(code, row["Title"], wage, occ_df, ind_df)

    ranked = sorted(RIASEC_COLUMNS, key=lambda c: (-user_scores[c], RIASEC_COLUMNS.index(c)))[:3]
    occ_top = sorted(RIASEC_COLUMNS, key=lambda c: (-row[c], RIASEC_COLUMNS.index(c)))[:3]
    shared = [c for c in occ_top if c in ranked]
    reasons = [(f"Your top interests ({', '.join(ranked)}) overlap with this career's main "
                f"dimensions: {', '.join(shared)}.") if shared else
               "This career's top interest dimensions differ from your top three, so treat it as exploratory."]
    found = [s["name"] for s in skills if s["status"] == "found"]
    if found:
        reasons.append(f"Your resume mentions {len(found)} of {len(skills)} skills listed for this career: "
                       f"{', '.join(found[:5])}.")

    return _clean({
        "code": code, "title": row["Title"], "description": row["Description"],
        "match": _cos(uvec, ovec), "reasons": reasons,
        "has_resume": resume_text is not None,
        "skills": skills, "salary": salary,
        "wage_ref": {"median": wage.get("malaysia_median_wage"), "date": wage.get("wage_data_date")},
        "outlook": _outlook(occ_df, salary["group"], hot),
        "related": _related(code, ovec, uvec),
    })