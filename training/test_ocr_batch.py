
from pathlib import Path
import io
import time

import fitz
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "resumee" / "Resumes PDF"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

MAX_PDFS = 10


def extract_text(pdf_path):
    with fitz.open(pdf_path) as doc:
        page_texts = []

        for page in doc:
            text = page.get_text("text").strip()

            if len(text) < 30:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )
                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )
                text = pytesseract.image_to_string(
                    image, lang="eng"
                ).strip()

            page_texts.append(text)

        return "\n".join(page_texts).strip()


def main():
    pdfs = []

    for category in sorted(DATASET.iterdir()):
        if not category.is_dir():
            continue

        category_pdfs = sorted(
            p for p in category.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        )

        remaining = MAX_PDFS - len(pdfs)
        pdfs.extend(category_pdfs[:remaining])

        if len(pdfs) >= MAX_PDFS:
            break

    print(f"Testing {len(pdfs)} PDFs...")
    start = time.time()

    for index, pdf_path in enumerate(pdfs, start=1):
        try:
            text = extract_text(pdf_path)
            print(
                f"[{index}/{len(pdfs)}] {pdf_path.name}: "
                f"{len(text)} characters"
            )
        except Exception as exc:
            print(
                f"[{index}/{len(pdfs)}] ERROR {pdf_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(f"\nElapsed time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
