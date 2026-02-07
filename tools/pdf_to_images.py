"""Simple PDF to images converter (PyMuPDF).

Install: pip install pymupdf
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF pages to images.")
    parser.add_argument("--input", required=True, help="PDF file or folder")
    parser.add_argument("--out-dir", required=True, help="Output folder")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI")
    parser.add_argument("--format", default="jpg", help="jpg/png/webp")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search PDFs recursively when input is a folder",
    )
    return parser.parse_args()


def _list_pdfs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(p for p in input_path.glob(pattern) if p.suffix.lower() == ".pdf")


def _convert_pdf(pdf_path: Path, out_dir: Path, dpi: int, fmt: str) -> int:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf") from exc

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    for idx in range(doc.page_count):
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out_path = out_dir / f"{pdf_path.stem}_p{idx + 1:04d}.{fmt}"
        pix.save(str(out_path))

    return doc.page_count


def main() -> None:
    args = _parse_args()
    fmt = args.format.lower().lstrip(".")
    if fmt not in {"jpg", "jpeg", "png", "webp"}:
        raise ValueError("--format must be jpg, jpeg, png, or webp")

    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    pdfs = _list_pdfs(input_path, args.recursive)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found at: {input_path}")

    total_pages = 0
    for pdf_path in pdfs:
        saved = _convert_pdf(pdf_path, out_dir, args.dpi, fmt)
        total_pages += saved
        print(f"[done] {pdf_path.name}: {saved} pages")

    print(f"[summary] total pages saved: {total_pages}")


if __name__ == "__main__":
    main()
