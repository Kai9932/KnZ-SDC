
"""
K&Z Aiversecity — Prepare Resume Dataset

Run from the project root:
    python training/prepare_dataset.py

Install dependencies if needed:
    python -m pip install pymupdf pandas
"""

import argparse
import hashlib
import re
from pathlib import Path
from collections import Counter

import fitz
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "resumee" / "Resumes PDF"
OUTPUT_DIR = PROJECT_ROOT / "results" / "resume_training"

MIN_TEXT_CHARS = 80

# Only merge category names that are reasonably clear equivalents.
ALIASES = {
    "accountant resumes": "Accountant",
    "advocate resumes": "Advocate",
    "agricultural resumes": "Agricultural",
    "apparel resumes": "Apparel",
    "architects resumes": "Architect",
    "arts resumes": "Arts",
    "aviation resumes": "Aviation",
    "banking resumes": "Banking",
    "data science": "Data Science",
    "data science resumes": "Data Science",
    "datascience": "Data Science",
    "business analyst resumes": "Business Analyst",
    "businessanalyst": "Business Analyst",
    "devops engineer resumes": "DevOps Engineer",
    "devopsengineer": "DevOps Engineer",
    "database resumes": "Database",
    "digital media resumes": "Digital Media",
    "education resumes": "Education",
    "finance resumes": "Finance",
    "hr resumes": "Human Resources",
    "human resources": "Human Resources",
    "information technology resumes": "Information Technology",
    "mechanical engineer resumes": "Mechanical Engineer",
    "mechanicalengineer": "Mechanical Engineer",
    "python developer resumes": "Python Developer",
    "pythondeveloper": "Python Developer",
    "react developer resumes": "React Developer",
    "sap developer resumes": "SAP Developer",
    "sapdeveloper": "SAP Developer",
    "sales resumes": "Sales",
    "testing resumes": "Testing",
    "consultant resumes": "Consultant",
    "et l developer resumes": "ETL Developer",
    "etl developer resumes": "ETL Developer",
    "web designing resumes": "Web Designing",
    "sql developer resumes": "SQL Developer",
}

# These names need human review rather than automatic merging.
REVIEW_LABELS = {
    "consult", "dot", "nse", "pbo", "bpo resumes",
    "pmo", "pmo resumes", "public",
    "building _construction resumes", "managment resumes",
    "designing resumes",
}


def canonical_label(folder_name):
    """Standardise obvious category-name variants."""
    key = re.sub(r"\s+", " ", folder_name.strip()).casefold()
    return ALIASES.get(
        key,
        re.sub(r"\s+", " ", folder_name.strip())
    )


def extract_pdf_text(pdf_path):
    """Extract embedded text; flag short/scanned PDFs for later OCR."""
    try:
        with fitz.open(pdf_path) as document:
            text = "\n".join(page.get_text("text") for page in document)

        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) < MIN_TEXT_CHARS:
            return text, "needs_ocr_or_review"

        return text, "ok"

    except Exception as error:
        return "", f"pdf_error:{type(error).__name__}"


def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8", errors="replace")
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the root folder containing career-category folders."
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}"
        )

    categories = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir()],
        key=lambda p: p.name.casefold()
    )

    if not categories:
        raise ValueError("No category folders were found.")

    records = []
    review_records = []
    category_counts = Counter()
    seen_hashes = {}
    duplicate_count = 0
    total_pdfs = 0

    print(f"Dataset: {dataset_root}")
    print(f"Category folders: {len(categories)}")
    print("Extracting PDF text. OCR is not run in this step.\n")

    for category_dir in categories:
        source_label = category_dir.name.strip()
        label = canonical_label(source_label)
        needs_label_review = (
            source_label.casefold() in REVIEW_LABELS
        )

        pdf_files = sorted(
            p for p in category_dir.rglob("*")
            if p.is_file() and p.suffix.casefold() == ".pdf"
        )

        print(
            f"{source_label}: {len(pdf_files)} PDFs "
            f"-> {label}"
        )

        for pdf_path in pdf_files:
            total_pdfs += 1
            text, status = extract_pdf_text(pdf_path)

            if status != "ok":
                review_records.append({
                    "source_file": str(pdf_path),
                    "source_label": source_label,
                    "canonical_label": label,
                    "status": status,
                    "text_characters": len(text),
                })
                continue

            digest = text_hash(text)
            duplicate_of = seen_hashes.get(digest, "")

            if duplicate_of:
                duplicate_count += 1
            else:
                seen_hashes[digest] = str(pdf_path)

            records.append({
                "text": text,
                "label": label,
                "source_label": source_label,
                "source_file": str(pdf_path),
                "text_characters": len(text),
                "text_sha256": digest,
                "duplicate_of": duplicate_of,
                "needs_label_review": needs_label_review,
            })

            category_counts[label] += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_output = OUTPUT_DIR / "resume_dataset.csv"
    review_output = OUTPUT_DIR / "label_review.csv"
    counts_output = OUTPUT_DIR / "category_counts.csv"

    pd.DataFrame(records).to_csv(
        dataset_output, index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        review_records,
        columns=[
            "source_file", "source_label", "canonical_label",
            "status", "text_characters"
        ]
    ).to_csv(review_output, index=False, encoding="utf-8-sig")

    pd.DataFrame([
        {"label": label, "count": count}
        for label, count in sorted(category_counts.items())
    ]).to_csv(counts_output, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)
    print(f"PDFs discovered:             {total_pdfs}")
    print(f"PDFs with extractable text:  {len(records)}")
    print(f"PDFs needing review/OCR:     {len(review_records)}")
    print(f"Repeated text copies:        {duplicate_count}")
    print(f"Canonical categories:        {len(category_counts)}")
    print(f"\nPrepared dataset: {dataset_output}")
    print(f"Review report:    {review_output}")
    print(f"Category counts:  {counts_output}")

    print("\nTop categories:")
    for label, count in category_counts.most_common(20):
        print(f"  {label:<35} {count:>5}")

    print("\nPreparation complete. No model was trained.")
    print("Review labels and duplicates before training.")


if __name__ == "__main__":
    main()
