
from pathlib import Path
import io

import fitz
import pytesseract
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "resumee" / "Resumes PDF"

# Explicit path: PATH configuration is not required.
TESSERACT_EXE = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


def main():
    print("Tesseract file exists:", Path(TESSERACT_EXE).is_file())
    print("Dataset exists:", DATASET.is_dir())

    # Verify Tesseract directly through pytesseract.
    try:
        print("Tesseract version:", pytesseract.get_tesseract_version())
        print("Languages:", pytesseract.get_languages(config=""))
    except Exception as exc:
        print("Tesseract check failed:", repr(exc))
        return

    pdfs = []
    for category in sorted(DATASET.iterdir()):
        if category.is_dir():
            matches = sorted(
                p for p in category.rglob("*")
                if p.is_file() and p.suffix.lower() == ".pdf"
            )
            if matches:
                pdfs.append(matches[0])
        if len(pdfs) >= 3:
            break

    for pdf_path in pdfs:
        print("\n" + "=" * 60)
        print("PDF:", pdf_path.name)
        print("Category:", pdf_path.parent.name)

        try:
            with fitz.open(pdf_path) as doc:
                page = doc[0]
                embedded = page.get_text("text").strip()
                print("Embedded characters:", len(embedded))

                if len(embedded) >= 30:
                    text = embedded
                    method = "embedded PDF text"
                else:
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
                    method = "Tesseract OCR"

                print("Method:", method)
                print("Extracted characters:", len(text))
                print("Text sample:")
                print(text[:700] if text else "[No text extracted]")

        except Exception as exc:
            print("PDF/OCR error:", type(exc).__name__, str(exc))

    print("\nDiagnostic finished; no training performed.")


if __name__ == "__main__":
    main()
