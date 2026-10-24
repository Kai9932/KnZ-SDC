"""K&Z Aiversecity — Resume dataset audit.

Run from the project root:
    python training/audit_dataset.py
Optional custom path:
    python training/audit_dataset.py --dataset "resumee/Resumes PDF"

This script audits folders and PDF files; it does not train a model or upload data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_DATASET = Path("resumee") / "Resumes PDF"
DEFAULT_REPORT_DIR = Path("results") / "resume_audit"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            while chunk := file.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    candidate = Path(__file__).resolve().parent.parent / path
    return candidate.resolve()


def audit_dataset(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Expected a folder like 'resumee/Resumes PDF'. Use --dataset to set the correct path."
        )
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {dataset_dir}")

    category_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold()
    )
    if not category_dirs:
        raise ValueError(f"No category subfolders found in {dataset_dir}")

    rows = []
    hashes = defaultdict(list)
    total_pdfs = total_other = total_empty = total_unreadable = 0

    print("\n" + "=" * 68)
    print("K&Z AIVERSECITY — RESUME DATASET AUDIT")
    print("=" * 68)
    print(f"Dataset: {dataset_dir}")
    print(f"Category folders: {len(category_dirs)}\n")
    print(f"{'CATEGORY':<34} {'PDFs':>6} {'OTHER':>7} {'STATUS':>13}")
    print("-" * 68)

    for category in category_dirs:
        files = [p for p in category.rglob("*") if p.is_file()]
        pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
        other = [p for p in files if p.suffix.lower() != ".pdf"]
        empty_count = unreadable_count = 0
        total_bytes = 0

        for pdf in pdfs:
            try:
                size = pdf.stat().st_size
                total_bytes += size
                if size == 0:
                    empty_count += 1
                    continue
            except OSError:
                unreadable_count += 1
                continue
            digest = sha256_file(pdf)
            if digest is None:
                unreadable_count += 1
            else:
                hashes[digest].append(str(pdf))

        status = "NO PDFs" if not pdfs else (
            "CHECK FILES" if empty_count or unreadable_count else "OK"
        )
        rows.append({
            "category_folder": category.name,
            "pdf_count": len(pdfs),
            "non_pdf_file_count": len(other),
            "empty_pdf_count": empty_count,
            "unreadable_file_count": unreadable_count,
            "total_pdf_size_mb": round(total_bytes / (1024 * 1024), 2),
            "status": status,
        })
        total_pdfs += len(pdfs)
        total_other += len(other)
        total_empty += empty_count
        total_unreadable += unreadable_count
        print(f"{category.name[:33]:<34} {len(pdfs):>6} {len(other):>7} {status:>13}")

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    summary = {
        "dataset_path": str(dataset_dir),
        "category_count": len(category_dirs),
        "total_pdfs": total_pdfs,
        "total_non_pdf_files": total_other,
        "total_empty_pdfs": total_empty,
        "total_unreadable_files": total_unreadable,
        "categories_without_pdfs": sum(row["pdf_count"] == 0 for row in rows),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_copies": sum(len(group) - 1 for group in duplicate_groups),
    }

    print("-" * 68)
    print(f"{'TOTAL':<34} {total_pdfs:>6} {total_other:>7}")
    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title():<30} {value}")
    if duplicate_groups:
        print("\nPossible duplicate PDFs (identical file contents):")
        for group in duplicate_groups[:10]:
            print("  - " + " | ".join(group[:4]))
            if len(group) > 4:
                print(f"    ... and {len(group) - 4} more copies")
        if len(duplicate_groups) > 10:
            print(f"  ... plus {len(duplicate_groups) - 10} duplicate groups.")
    print("\nNOTES")
    print("• This audit counts files; it does not assess resume quality.")
    print("• Review variants such as 'Data Science' and 'data science resumes'.")
    print("• Use only authorised data and protect personal information.")
    print("• Evaluate a trained classifier with macro F1, per-class scores, and a confusion matrix.")
    return rows, summary


def write_reports(rows, summary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "category_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("K&Z Aiversecity — Resume Dataset Audit\n")
        file.write("=" * 42 + "\n")
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")
    print(f"\nReports saved to:\n  {csv_path}\n  {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit resume PDF dataset folders.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help='Dataset root (default: "resumee/Resumes PDF").')
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR,
                        help='Report folder (default: "results/resume_audit").')
    args = parser.parse_args()
    dataset = resolve_path(args.dataset)
    output = args.output
    if not output.is_absolute():
        output = Path(__file__).resolve().parent.parent / output
    try:
        rows, summary = audit_dataset(dataset)
        write_reports(rows, summary, output)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        return 1
    print("\nAudit complete. No model training was performed yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
