
from retriever import retrieve_documents
from llm_generator import generate_answer


def answer_question(
    question,
    save_path,
    top_k=4,
    source_name="Uploaded PDF"
):

    retrieved_documents = retrieve_documents(
        query=question,
        save_path=save_path,
        top_k=top_k
    )

    if not retrieved_documents:
        return {
            "answer": "No relevant information was found in the document.",
            "sources": []
        }

    for document in retrieved_documents:
        document["source"] = source_name

    answer = generate_answer(
        question=question,
        retrieved_documents=retrieved_documents
    )

    sources = [
        {
            "source": document["source"],
            "page_number": document["page_number"],
            "text": document["text"],
            "score": document["score"]
        }
        for document in retrieved_documents
    ]

    return {
        "answer": answer,
        "sources": sources
    }
