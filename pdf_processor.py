
from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF.
    Returns text page by page with page numbers.
    """

    extracted_pages = []

    try:
        reader = PdfReader(pdf_file)

        if reader.is_encrypted:
            raise ValueError("The PDF is password protected.")

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text and text.strip():

                extracted_pages.append({
                    "page_number": page_number,
                    "text": text.strip()
                })

        if not extracted_pages:
            raise ValueError(
                "No readable text found in this PDF."
            )

        return extracted_pages

    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")


def get_document_text(extracted_pages):
    """
    Combine extracted page text into one document.
    """

    full_text = ""

    for page in extracted_pages:

        full_text += (
            f"\n\nPage {page['page_number']}\n"
            f"{page['text']}"
        )

    return full_text
