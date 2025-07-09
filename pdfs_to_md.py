from pathlib import Path

from docling.document_converter import DocumentConverter
from icecream import ic  # type: ignore

def convert_pdfs_to_md(pdf_dir: Path, output_dir: Path):
    """
    Convert all PDF files in the specified directory to Markdown format.
    
    Args:
        pdf_dir (Path): Directory containing PDF files.
        output_dir (Path): Directory where the converted Markdown files will be saved.
    """
    converter = DocumentConverter()
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        ic("Converting PDF:", pdf_file)
        md_file = output_dir / f"{pdf_file.stem}.md"
        result = converter.convert(pdf_file)
        md_file.write_text(result.document.export_to_markdown())
        ic("Converted PDF:", pdf_file, "to", md_file)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python pdfs_to_md.py <pdf_directory> <output_directory>")
        sys.exit(1)
    
    pdf_directory = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])

    convert_pdfs_to_md(pdf_directory, output_directory)
