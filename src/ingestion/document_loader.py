"""Load text from Markdown and PDF documents."""

from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".md", ".pdf"}


def load_document(file_path: str | Path) -> str:
    """Load a Markdown or PDF document and return its text."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]
        return "\n\n".join(pages)

    raise ValueError(
        f"Unsupported file type: {path.suffix}. "
        "Supported types are .md and .pdf."
    )