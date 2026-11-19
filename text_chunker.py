from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(extracted_pages):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    for page in extracted_pages:

        chunks = text_splitter.split_text(
            page["text"]
        )

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "page_number": page["page_number"]
                }
            })

    return all_chunks