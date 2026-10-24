from pathlib import Path
from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "processed").is_dir()
    else SCRIPT_DIR.parent
)
BASE_DIR = PROJECT_DIR
PROCESSED_DIR = PROJECT_DIR / "processed"

DATA_PATH = PROCESSED_DIR / "career_occupation_master.csv"
ESSENTIAL_SKILLS_PATH = PROCESSED_DIR / "essential_skills_clean.csv"
SOFTWARE_SKILLS_PATH = PROCESSED_DIR / "software_skills_clean.csv"
WAGES_PATH = PROCESSED_DIR / "malaysia_wages_clean.csv"

RIASEC_COLUMNS = [
    "Realistic",
    "Investigative",
    "Artistic",
    "Social",
    "Enterprising",
    "Conventional",
]

RIASEC_LETTERS = {
    "Realistic": "R",
    "Investigative": "I",
    "Artistic": "A",
    "Social": "S",
    "Enterprising": "E",
    "Conventional": "C",
}

QUESTION_MAP = {
    "Realistic": [1, 7, 14, 22, 30, 32, 37],
    "Investigative": [2, 11, 18, 21, 26, 33, 39],
    "Artistic": [8, 17, 23, 27, 31, 41, 42],
    "Social": [4, 12, 13, 20, 28, 40, 3],
    "Enterprising": [5, 10, 16, 19, 29, 36, 34],
    "Conventional": [6, 9, 15, 24, 25, 35, 38],
}

REQUIRED_OCCUPATION_COLUMNS = [
    "O*NET-SOC Code",
    "Title",
    "Description",
    *RIASEC_COLUMNS,
]

MIN_WAGE_TITLE_SIMILARITY = 0.82

TOP_ESSENTIAL_SKILLS = 5
TOP_SOFTWARE_SKILLS = 5


def read_csv_if_available(path):
    """Load a CSV, returning None if it is an optional missing file."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"Warning: Could not read {path.name}: {exc}")
        return None


def normalise_text(value):
    """Normalise a title or label for comparison."""
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_number(value):
    """Convert a value to a finite float, or NaN if invalid."""
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError):
        return np.nan


def format_skill_list(items):
    """Format a list of skill descriptions for output."""
    cleaned = []

    for item in items:
        if pd.isna(item):
            continue

        item = str(item).strip()

        if item and item not in cleaned:
            cleaned.append(item)

    return "; ".join(cleaned) if cleaned else "No matching data available"



QUESTIONS = {
    1: "I like to work on cars",
    2: "I like to do puzzles",
    3: "I am good at working independently",
    4: "I like to work in teams",
    5: "I am ambitious and set goals for myself",
    6: "I like to organise files, desks, or offices",
    7: "I like to build things",
    8: "I like to read about art and music",
    9: "I like to have clear instructions to follow",
    10: "I like to influence or persuade people",
    11: "I like to do experiments",
    12: "I like to teach or train people",
    13: "I like helping people solve their problems",
    14: "I like to take care of animals",
    15: "I wouldn't mind working eight hours a day in an office",
    16: "I like selling things",
    17: "I enjoy creative writing",
    18: "I enjoy science",
    19: "I am quick to take on new responsibilities",
    20: "I am interested in healing people",
    21: "I enjoy figuring out how things work",
    22: "I like assembling things",
    23: "I am a creative person",
    24: "I pay attention to details",
    25: "I like filing or typing",
    26: "I like analysing problems or situations",
    27: "I like playing instruments or singing",
    28: "I enjoy learning about other cultures",
    29: "I would like to start my own business",
    30: "I like to cook",
    31: "I like acting in plays",
    32: "I am a practical person",
    33: "I like working with numbers or charts",
    34: "I like discussions about issues",
    35: "I am good at keeping records",
    36: "I like to lead",
    37: "I like working outdoors",
    38: "I would like to work in an office",
    39: "I am good at mathematics",
    40: "I like helping people",
    41: "I like to draw",
    42: "I like giving speeches",
}


def score_riasec(answers):
    """
    Score 42 Y/N answers.

    Returns:
        scores: dictionary of six raw scores from 0 to 7
        interest_code: three-letter code, or None if tied at
                       the third-place cutoff
    """
    if set(answers) != set(range(1, 43)):
        raise ValueError("Exactly 42 answers are required (questions 1-42).")

    normalised = {}

    for question, answer in answers.items():
        if isinstance(answer, (bool, np.bool_)):
            normalised[question] = bool(answer)
        elif isinstance(answer, str) and answer.strip().upper() in {"Y", "N"}:
            normalised[question] = answer.strip().upper() == "Y"
        else:
            raise ValueError(
                f"Question {question}: answer must be Y or N."
            )

    scores = {
        category: sum(normalised[q] for q in question_numbers)
        for category, question_numbers in QUESTION_MAP.items()
    }

    ranked_scores = sorted(scores.values(), reverse=True)
    third_place_score = ranked_scores[2]

    above_cutoff = sum(
        score > third_place_score for score in scores.values()
    )
    tied_at_cutoff = sum(
        score == third_place_score for score in scores.values()
    )

    if above_cutoff < 3 and tied_at_cutoff > 1:
        interest_code = None
    else:
        ranked = sorted(
            RIASEC_COLUMNS,
            key=lambda category: (
                -scores[category],
                RIASEC_COLUMNS.index(category),
            ),
        )
        interest_code = "".join(
            RIASEC_LETTERS[category] for category in ranked[:3]
        )

    return scores, interest_code


def load_occupations():
    """Load and validate occupation-level RIASEC profiles."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Occupation dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing = [
        column for column in REQUIRED_OCCUPATION_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Occupation dataset is missing columns: {missing}"
        )

    df = df.copy()

    for column in RIASEC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["O*NET-SOC Code", "Title", *RIASEC_COLUMNS]
    ).copy()

    df = df.drop_duplicates(subset=["O*NET-SOC Code"])

    valid = (
        np.isfinite(df[RIASEC_COLUMNS].to_numpy(dtype=float)).all(axis=1)
        & (df[RIASEC_COLUMNS].to_numpy(dtype=float) >= 0).all(axis=1)
    )
    df = df.loc[valid].copy()

    df = df[df[RIASEC_COLUMNS].sum(axis=1) > 0].copy()

    if df.empty:
        raise ValueError("No usable occupation profiles were found.")

    return df.reset_index(drop=True)


def load_essential_skills():
    """
    Load essential skills.

    Expected columns include:
        O*NET-SOC Code, Element Name, importance, level
    """
    df = read_csv_if_available(ESSENTIAL_SKILLS_PATH)

    if df is None:
        return None

    required = {"O*NET-SOC Code", "Element Name"}
    if not required.issubset(df.columns):
        print("Warning: Essential skills dataset has unexpected columns.")
        return None

    df = df.copy()
    df["O*NET-SOC Code"] = df["O*NET-SOC Code"].astype(str).str.strip()
    df["Element Name"] = df["Element Name"].fillna("").astype(str).str.strip()

    for column in ("importance", "level"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def get_essential_skills(skills_df, occupation_code):
    """Return the most important skills for an occupation."""
    if skills_df is None:
        return "Essential skills dataset unavailable"

    rows = skills_df[
        skills_df["O*NET-SOC Code"] == str(occupation_code).strip()
    ].copy()

    rows = rows[rows["Element Name"].str.strip() != ""]

    if rows.empty:
        return "No matching essential skills available"

    sort_columns = []
    ascending = []

    if "importance" in rows.columns:
        sort_columns.append("importance")
        ascending.append(False)

    if "level" in rows.columns:
        sort_columns.append("level")
        ascending.append(False)

    if sort_columns:
        rows = rows.sort_values(sort_columns, ascending=ascending)

    return format_skill_list(
        rows["Element Name"].head(TOP_ESSENTIAL_SKILLS).tolist()
    )


def load_software_skills():
    """
    Load software skills.

    Expected columns include:
        O*NET-SOC Code, Element Name, Hot Technology, In Demand
    """
    df = read_csv_if_available(SOFTWARE_SKILLS_PATH)

    if df is None:
        return None

    required = {"O*NET-SOC Code", "Element Name"}
    if not required.issubset(df.columns):
        print("Warning: Software skills dataset has unexpected columns.")
        return None

    df = df.copy()
    df["O*NET-SOC Code"] = df["O*NET-SOC Code"].astype(str).str.strip()
    df["Element Name"] = df["Element Name"].fillna("").astype(str).str.strip()

    return df


def flag_is_true(value):
    """Handle boolean flags stored as bools, numbers, or text."""
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true", "yes", "y", "1", "hot", "in demand"
    }


def get_software_skills(skills_df, occupation_code):
    """Return relevant software skills, prioritising flagged skills."""
    if skills_df is None:
        return "Software skills dataset unavailable"

    rows = skills_df[
        skills_df["O*NET-SOC Code"] == str(occupation_code).strip()
    ].copy()

    rows = rows[rows["Element Name"].str.strip() != ""]

    if rows.empty:
        return "No matching software skills available"

    priority = pd.Series(0, index=rows.index, dtype=int)

    if "Hot Technology" in rows.columns:
        priority += rows["Hot Technology"].apply(flag_is_true).astype(int)

    if "In Demand" in rows.columns:
        priority += rows["In Demand"].apply(flag_is_true).astype(int)

    rows["_priority"] = priority
    rows = rows.sort_values(
        ["_priority", "Element Name"],
        ascending=[False, True],
    )

    descriptions = []

    for _, row in rows.iterrows():
        name = str(row["Element Name"]).strip()
        tags = []

        if "Hot Technology" in rows.columns and flag_is_true(
            row["Hot Technology"]
        ):
            tags.append("Hot technology")

        if "In Demand" in rows.columns and flag_is_true(
            row["In Demand"]
        ):
            tags.append("In demand")

        description = f"{name} ({', '.join(tags)})" if tags else name

        if description not in descriptions:
            descriptions.append(description)

        if len(descriptions) >= TOP_SOFTWARE_SKILLS:
            break

    return format_skill_list(descriptions)


def load_malaysia_wages():
    """
    Load Malaysian wage data.

    Expected columns include:
        date, sex, occupation, wages_median, wages_mean

    This loader does not assume that every row represents all sexes.
    """
    df = read_csv_if_available(WAGES_PATH)

    if df is None:
        return None

    required = {
        "occupation",
        "wages_median",
        "wages_mean",
    }

    if not required.issubset(df.columns):
        print("Warning: Malaysian wage dataset has unexpected columns.")
        return None

    df = df.copy()
    df["occupation"] = df["occupation"].fillna("").astype(str).str.strip()
    df["_normalised_occupation"] = df["occupation"].apply(normalise_text)

    for column in ("wages_median", "wages_mean"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "date" in df.columns:
        df["_date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["_date_parsed"] = pd.NaT

    return df

def wage_group_for_title(title):
    text = normalise_text(title)

    rules = [
        ("managers", ("manager", "director", "chief executive", "media technical director")),
        ("professionals", (
            "architect", "planner", "naturalist", "curator", "educator",
            "teacher", "engineer", "scientist", "analyst", "designer",
            "dietitian", "dietetic", "doctor", "physician", "nurse",
            "lawyer", "accountant", "counselor", "counsellor", "therapist",
            "researcher", "librarian", "artist", "writer", "urban and regional",
        )),
        ("technicians", ("technician", "technologist", "drafting", "surveying")),
        ("clerical", ("clerk", "secretary", "administrative assistant", "bookkeeping")),
        ("service_sales", ("tour guide", "sales", "retail", "waiter", "chef", "cook", "hospitality")),
        ("craft", ("electrician", "plumber", "carpenter", "mechanic", "welder")),
        ("operators", ("operator", "driver", "pilot", "machine")),
        ("agriculture", ("farmer", "agricultural", "forester", "landscape worker")),
        ("elementary", ("cleaner", "labourer", "laborer", "helper")),
    ]

    for group, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return group
    return None


def match_wage_occupation(wages_df, occupation_title):
    """
    Find a wage record using an exact or sufficiently similar title.

    Exact title matches are preferred. Fuzzy matches below the configured
    threshold are rejected to reduce the risk of misleading salary results.

    If sex-specific rows exist but no overall/total row exists, wages are
    left blank rather than arbitrarily selecting one sex.
    """
    empty_result = {
        "malaysia_median_wage": np.nan,
        "malaysia_mean_wage": np.nan,
        "wage_data_date": None,
        "wage_matched_occupation": None,
        "wage_title_similarity": np.nan,
    }

    if wages_df is None or wages_df.empty:
        return empty_result

    target = normalise_text(occupation_title)

    if not target:
        return empty_result

    group = wage_group_for_title(occupation_title)
    if group is not None:
        group_rows = wages_df[
            wages_df["_normalised_occupation"] == normalise_text(group)
        ].copy()
        if not group_rows.empty:
            if "sex" in group_rows.columns:
                sex_labels = group_rows["sex"].apply(normalise_text)
                overall = group_rows[sex_labels.isin({
                    "both sexes", "both", "total", "all", "overall", "all sexes"
                })]
                if not overall.empty:
                    group_rows = overall
                else:
                    return empty_result
            if "_date_parsed" in group_rows.columns:
                group_rows = group_rows.sort_values(
                    "_date_parsed", ascending=False, na_position="last"
                )
            row = group_rows.iloc[0]
            raw_date = row.get("date", None)
            date_label = None if pd.isna(raw_date) else str(raw_date)
            return {
                "malaysia_median_wage": safe_number(row.get("wages_median", np.nan)),
                "malaysia_mean_wage": safe_number(row.get("wages_mean", np.nan)),
                "wage_data_date": date_label,
                "wage_matched_occupation": row.get("occupation", group),
                "wage_title_similarity": 0.0,
            }

    candidates = wages_df[
        wages_df["_normalised_occupation"] != ""
    ].copy()

    if candidates.empty:
        return empty_result

    exact = candidates[
        candidates["_normalised_occupation"] == target
    ]

    if not exact.empty:
        matched = exact
        similarity = 1.0
    else:
        similarities = candidates["_normalised_occupation"].apply(
            lambda value: SequenceMatcher(None, target, value).ratio()
        )

        best_index = similarities.idxmax()
        similarity = float(similarities.loc[best_index])

        if similarity < MIN_WAGE_TITLE_SIMILARITY:
            return empty_result

        best_title = candidates.loc[best_index, "_normalised_occupation"]
        matched = candidates[
            candidates["_normalised_occupation"] == best_title
        ].copy()

    if "sex" in matched.columns:
        sex_normalised = matched["sex"].apply(normalise_text)

        overall_labels = {
            "both sexes",
            "both",
            "total",
            "all",
            "overall",
            "all sexes",
        }

        overall = matched[sex_normalised.isin(overall_labels)]

        if not overall.empty:
            matched = overall
        else:

            return empty_result

    if "_date_parsed" in matched.columns:
        matched = matched.sort_values(
            "_date_parsed",
            ascending=False,
            na_position="last",
        )

    row = matched.iloc[0]

    median = safe_number(row.get("wages_median", np.nan))
    mean = safe_number(row.get("wages_mean", np.nan))

    raw_date = row.get("date", None)
    if pd.isna(raw_date):
        date_label = None
    else:
        date_label = str(raw_date)

    return {
        "malaysia_median_wage": median,
        "malaysia_mean_wage": mean,
        "wage_data_date": date_label,
        "wage_matched_occupation": row.get("occupation", None),
        "wage_title_similarity": similarity,
    }



def explain_match(user_scores, occupation_row):
    """
    Explain the highest shared RIASEC interests.

    Scores are used to describe profile alignment, not job suitability,
    qualifications, or likelihood of career success.
    """
    occupation_scores = {
        category: safe_number(occupation_row[category])
        for category in RIASEC_COLUMNS
    }

    ranked_user = sorted(
        RIASEC_COLUMNS,
        key=lambda category: (
            -user_scores[category],
            RIASEC_COLUMNS.index(category),
        ),
    )

    top_user_score = max(user_scores.values())
    strongest_user = [
        category for category in ranked_user
        if user_scores[category] == top_user_score
    ]

    if len(strongest_user) == len(RIASEC_COLUMNS):
        user_summary = (
            "Your answers show equal interest across all six RIASEC "
            "categories, so this test does not identify a clear preference."
        )
    else:
        user_summary = (
            "Your strongest interest area(s): "
            + ", ".join(strongest_user)
            + "."
        )
    occupation_top = sorted(
        RIASEC_COLUMNS,
        key=lambda category: (
            -occupation_scores[category],
            RIASEC_COLUMNS.index(category),
        ),
    )[:3]

    user_top = ranked_user[:3]
    shared = [category for category in occupation_top if category in user_top]

    if shared:
        alignment = (
            "Shared top interest dimensions: " + ", ".join(shared) + "."
        )
    else:
        alignment = (
            "The occupation's top three interest dimensions differ from "
            "your top three dimensions."
        )

    return f"{user_summary} {alignment}"

def match_careers(user_scores, top_n=10):
    """
    Rank occupations using cosine similarity and enrich the results with
    skills, explanations, and available Malaysian wage information.

    Returns a pandas DataFrame suitable for display or CSV export.
    """
    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer.")

    if set(user_scores) != set(RIASEC_COLUMNS):
        raise ValueError("Provide scores for all six RIASEC categories.")

    user_vector = np.array(
        [user_scores[category] for category in RIASEC_COLUMNS],
        dtype=float,
    )

    if not np.isfinite(user_vector).all():
        raise ValueError("RIASEC scores must be finite numbers.")

    if ((user_vector < 0) | (user_vector > 7)).any():
        raise ValueError("Each raw RIASEC score must be between 0 and 7.")

    if user_vector.sum() == 0:
        raise ValueError(
            "All RIASEC scores are zero. Complete the questionnaire first."
        )

    occupations = load_occupations()

    occupation_vectors = occupations[
        RIASEC_COLUMNS
    ].to_numpy(dtype=float, copy=True)

    row_norms = np.linalg.norm(
        occupation_vectors,
        axis=1,
        keepdims=True,
    )

    valid_norms = row_norms[:, 0] > 0
    occupations = occupations.loc[valid_norms].copy()
    occupation_vectors = occupation_vectors[valid_norms]
    row_norms = row_norms[valid_norms]

    if len(occupations) == 0:
        raise ValueError("No valid occupation profiles are available.")

    flat_profile = len(np.unique(user_vector)) == 1

    if flat_profile:
        occupations["match_score"] = np.nan
    else:
        normalised_user = user_vector / np.linalg.norm(user_vector)
        normalised_occupations = occupation_vectors / row_norms
        similarities = normalised_occupations @ normalised_user
        occupations["match_score"] = np.clip(similarities * 100, 0, 100)

    essential_skills = load_essential_skills()
    software_skills = load_software_skills()
    wages = load_malaysia_wages()

    ranked_user = sorted(
        RIASEC_COLUMNS,
        key=lambda category: (
            -user_scores[category],
            RIASEC_COLUMNS.index(category),
        ),
    )

    top_user_score = max(user_scores.values())
    strongest = [
        category for category in ranked_user
        if user_scores[category] == top_user_score
    ]

    profile_summary = ", ".join(strongest)

    occupations["your_interest_profile"] = profile_summary

    occupations["occupation_top_interests"] = occupations.apply(
        lambda row: ", ".join(
            sorted(
                RIASEC_COLUMNS,
                key=lambda category: (
                    -row[category],
                    RIASEC_COLUMNS.index(category),
                ),
            )[:3]
        ),
        axis=1,
    )

    occupations["match_explanation"] = occupations.apply(
        lambda row: explain_match(user_scores, row),
        axis=1,
    )

    if flat_profile:
        results = occupations.sort_values("Title", ascending=True).head(top_n).copy()
    else:
        results = occupations.sort_values(
            "match_score",
            ascending=False,
        ).head(top_n).copy()

    results["important_skills"] = results.apply(
        lambda row: get_essential_skills(
            essential_skills,
            row["O*NET-SOC Code"],
        ),
        axis=1,
    )

    results["related_software_skills"] = results.apply(
        lambda row: get_software_skills(
            software_skills,
            row["O*NET-SOC Code"],
        ),
        axis=1,
    )

    wage_results = results.apply(
        lambda row: match_wage_occupation(wages, row["Title"]),
        axis=1,
    )

    wage_df = pd.DataFrame(wage_results.tolist(), index=results.index)
    results = pd.concat([results, wage_df], axis=1)

    output_columns = [
        "O*NET-SOC Code",
        "Title",
        "Description",
        "match_score",
        "your_interest_profile",
        "occupation_top_interests",
        "match_explanation",
        "important_skills",
        "related_software_skills",
        "malaysia_median_wage",
        "malaysia_mean_wage",
        "wage_data_date",
        "wage_matched_occupation",
        "wage_title_similarity",
    ]

    return results[output_columns].reset_index(drop=True)



def display_results(results):
    """Print enriched career results in the terminal."""
    print("\n========== TOP CAREER MATCHES ==========")

    for index, row in results.iterrows():
        print("\n" + "=" * 60)
        print(f"{index + 1}. {row['Title']}")
        print("=" * 60)

        print(f"O*NET code: {row['O*NET-SOC Code']}")
        if pd.notna(row["match_score"]):
            print(f"RIASEC profile similarity: {row['match_score']:.1f}%")
        else:
            print("RIASEC profile similarity: Not ranked (all six interests tied)")
        print(f"Your strongest interest area(s): {row['your_interest_profile']}")
        print(f"Occupation's top interests: {row['occupation_top_interests']}")

        print("\nOccupation description:")
        print(row["Description"])

        print("\nWhy it matches:")
        print(row["match_explanation"])

        print("\nImportant skills:")
        print(row["important_skills"])

        print("\nRelated software skills:")
        print(row["related_software_skills"])

        median = row["malaysia_median_wage"]
        mean = row["malaysia_mean_wage"]

        print("\nMalaysian wage information:")

        if pd.notna(median):
            print(f"Median wage: RM {median:,.2f}")
        else:
            print("Median wage: No reliable matching record available")

        if pd.notna(mean):
            print(f"Mean wage: RM {mean:,.2f}")
        else:
            print("Mean wage: No reliable matching record available")

        if row["wage_matched_occupation"]:
            print(
                "Matched wage occupation: "
                f"{row['wage_matched_occupation']}"
            )

        if row["wage_data_date"]:
            print(f"Wage data period: {row['wage_data_date']}")

        if pd.notna(row["wage_title_similarity"]):
            if row["wage_title_similarity"] == 0:
                print("Wage basis: broad Malaysian occupational-group reference")
            else:
                print(
                    "Wage title similarity: "
                    f"{row['wage_title_similarity'] * 100:.1f}%"
                )

    print(
        "\nNote: RIASEC similarity is not a probability of career success. "
        "If all six interests are tied, careers are listed alphabetically "
        "for exploration rather than ranked by a misleading similarity score. "
        "Wage figures from broad occupational groups are references, not "
        "career-specific salary estimates."
    )



def main():
    print("\n" + "=" * 48)
    print("       CAREER GUIDANCE - RIASEC TEST")
    print("=" * 48)
    print("Answer Y for Agree or N for Do not agree.")

    answers = {}

    for number, statement in QUESTIONS.items():
        while True:
            answer = input(
                f"\nQ{number:02d}/42: {statement}\nYour answer (Y/N): "
            ).strip().upper()

            if answer in {"Y", "N"}:
                answers[number] = answer
                break

            print("Invalid answer. Please enter Y or N.")

    try:
        scores, code = score_riasec(answers)

        print("\n========== YOUR RIASEC SCORES ==========")

        for category in RIASEC_COLUMNS:
            print(
                f"{RIASEC_LETTERS[category]} - {category:<15}: "
                f"{scores[category]}/7"
            )

        if code:
            print(f"\nYour three-letter interest code: {code}")
        else:
            print(
                "\nYour RIASEC scores are tied at the three-letter-code cutoff."
                "\nNo unique three-letter code can be assigned."
            )

        if len(set(scores.values())) == 1:
            print(
                "\nIMPORTANT: All six interest scores are identical. "
                "The questionnaire has not identified a distinct interest profile."
                "\nThe career list below is exploratory only. Retake the test "
                "and answer based on your actual preferences rather than choosing Y for every item."
            )
        elif sum(score == max(scores.values()) for score in scores.values()) >= 4:
            print(
                "\nNOTE: Several interest categories share the highest score. "
                "Treat the ranking as exploratory; the test has limited ability "
                "to distinguish your preferences."
            )

        results = match_careers(scores, top_n=10)
        display_results(results)

        output_path = BASE_DIR / "career_match_results.csv"
        results.to_csv(output_path, index=False)

        print(f"\nResults saved to: {output_path}")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
    except (OSError, pd.errors.ParserError) as error:
        print(f"\nDataset error: {error}")


if __name__ == "__main__":
    main()