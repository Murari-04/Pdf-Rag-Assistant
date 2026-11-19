import os
import uuid
import hashlib
import streamlit as st

from pdf_processor import extract_text_from_pdf, get_document_text
from text_chunker import split_documents
from vector_store import create_vector_store
from rag import answer_question

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="PDF RAG Assitant",
    page_icon="📚",
    layout="wide"
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_FILES = 5


# ---------------- SESSION STATE ----------------

defaults = {
    "session_id": str(uuid.uuid4()),
    "chat_history": {},
    "processed_documents": {},
    "pending_question": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- HEADER ----------------

st.title("📚 PDF RAG Assistant")
st.markdown("### Your Intelligent PDF Question-Answering Assistant")

st.write(
    "Upload any supported text-based PDF and ask questions "
    "about its contents using Google Gemini, LangChain and FAISS."
)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "📂 Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = {}
        st.session_state.pending_question = None
        st.success("Chat history cleared!")
        st.rerun()

    st.caption("Powered by Python, Streamlit, Gemini, LangChain and FAISS.")


# ---------------- DOCUMENT PROCESSING ----------------

available_documents = {}

if uploaded_files:

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Maximum {MAX_FILES} PDFs are allowed per upload.")

    for uploaded_file in uploaded_files[:MAX_FILES]:

        try:
            pdf_bytes = uploaded_file.getvalue()

            if not pdf_bytes:
                st.warning(f"{uploaded_file.name} is empty.")
                continue

            if len(pdf_bytes) > MAX_FILE_SIZE:
                st.warning(f"{uploaded_file.name} exceeds the 20 MB limit.")
                continue

            if b"%PDF-" not in pdf_bytes[:1024]:
                st.warning(f"{uploaded_file.name} is not a valid PDF.")
                continue

            # Unique document identification
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
            document_key = f"document_{pdf_hash}"

            # Session-specific FAISS storage
            save_path = os.path.join(
                "vectorstore",
                st.session_state.session_id,
                document_key
            )

            index_path = os.path.join(save_path, "index.faiss")
            metadata_path = os.path.join(save_path, "index.pkl")

            st.subheader(f"📄 {uploaded_file.name}")

            # Reuse previously extracted document data
            cached = st.session_state.processed_documents.get(pdf_hash)

            if cached:
                pages = cached["pages"]
                full_text = cached["full_text"]
                chunks = cached["chunks"]

            else:
                with st.spinner(f"Processing {uploaded_file.name}..."):

                    uploaded_file.seek(0)
                    pages = extract_text_from_pdf(uploaded_file)

                    if not pages:
                        st.warning("No readable pages found.")
                        continue

                    full_text = get_document_text(pages)

                    if not full_text or not full_text.strip():
                        st.warning(
                            "No extractable text found. "
                            "OCR may be required for scanned PDFs."
                        )
                        continue

                    chunks = split_documents(pages)

                    if not chunks:
                        st.warning("No text chunks generated.")
                        continue

                st.session_state.processed_documents[pdf_hash] = {
                    "pages": pages,
                    "full_text": full_text,
                    "chunks": chunks
                }

            # Document information
            with st.expander(f"📊 Document Information - {uploaded_file.name}"):

                col1, col2, col3 = st.columns(3)

                col1.metric("Pages", len(pages))
                col2.metric("Characters", len(full_text))
                col3.metric("Chunks", len(chunks))

                with st.expander("📖 View Extracted Text"):
                    st.text(full_text[:10000])

                with st.expander("🔍 View Generated Chunks"):
                    for i, chunk in enumerate(chunks):
                        page = chunk.get("metadata", {}).get(
                            "page_number", "Unknown"
                        )
                        st.markdown(f"**Chunk {i + 1} | Page {page}**")
                        st.write(chunk["text"])
                        st.divider()

            # Create or reuse FAISS database
            database_exists = (
                os.path.isfile(index_path)
                and os.path.isfile(metadata_path)
            )

            if database_exists:
                st.success("FAISS database ready!")

            else:
                os.makedirs(save_path, exist_ok=True)

                with st.spinner("Generating embeddings and creating FAISS database..."):
                    vector_db = create_vector_store(
                        chunks,
                        save_path=save_path
                    )

                st.success("FAISS database created successfully!")

                col1, col2 = st.columns(2)
                col1.metric("Vectors", vector_db.index.ntotal)
                col2.metric("Dimensions", vector_db.index.d)

            # Register indexed document
            if (
                os.path.isfile(index_path)
                and os.path.isfile(metadata_path)
            ):
                available_documents[document_key] = {
                    "name": uploaded_file.name,
                    "path": save_path,
                    "pages": len(pages),
                    "chunks": len(chunks)
                }

        except Exception as error:
            print(f"Document processing error: {error}")
            st.error(
                f"Unable to process {uploaded_file.name}. "
                "Please check the PDF and try again."
            )


# ---------------- DOCUMENT DASHBOARD ----------------

st.divider()
st.header("📚 Your Documents")

if available_documents:

    col1, col2, col3 = st.columns(3)

    col1.metric("Indexed Documents", len(available_documents))
    col2.metric(
        "Total Pages",
        sum(doc["pages"] for doc in available_documents.values())
    )
    col3.metric(
        "Total Chunks",
        sum(doc["chunks"] for doc in available_documents.values())
    )

    for document in available_documents.values():
        st.markdown(f"📄 **{document['name']}**")
        st.caption(
            f"{document['pages']} pages | "
            f"{document['chunks']} chunks | FAISS ready"
        )

else:
    st.info("Upload PDF documents to activate DocuMind AI.")


# ---------------- CHATBOT ----------------

st.divider()
st.header("💬 Chat with Your Documents")

if available_documents:

    selected_key = st.selectbox(
        "Select a PDF document",
        options=list(available_documents.keys()),
        format_func=lambda key: available_documents[key]["name"],
        key="document_selector"
    )

    selected = available_documents[selected_key]
    document_name = selected["name"]
    save_path = selected["path"]

    st.success(f"Connected to: {document_name}")

    chat_history = st.session_state.chat_history.setdefault(
        selected_key, []
    )

    # Suggested questions
    if not chat_history:

        st.markdown("### 👋 Hello! How can I help you?")

        suggestions = [
            "Summarize this document.",
            "What are the key points?",
            "Explain the main concepts.",
            "What are the important findings?",
            "Explain this document in simple language."
        ]

        for i, suggestion in enumerate(suggestions):

            if st.button(
                f"💡 {suggestion}",
                key=f"suggest_{selected_key}_{i}",
                use_container_width=True
            ):
                st.session_state.pending_question = suggestion
                st.rerun()

    # Display conversation
    for message in chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):

                with st.expander("📚 View Sources and References"):

                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i + 1}**")
                        st.caption(
                            f"Document: {source.get('source', document_name)} | "
                            f"Page: {source.get('page_number', 'Unknown')}"
                        )
                        st.write(source.get("text", ""))
                        st.divider()

    # Question input
    question = st.chat_input("Ask anything about your PDF...")

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    # Generate response
    if question and question.strip():

        question = question.strip()

        chat_history.append({
            "role": "user",
            "content": question
        })

        try:
            with st.spinner("🔎 Searching and generating your answer..."):

                result = answer_question(
                    question=question,
                    save_path=save_path,
                    top_k=4,
                    source_name=document_name
                )

            chat_history.append({
                "role": "assistant",
                "content": result.get("answer", "No answer generated."),
                "sources": result.get("sources", [])
            })

        except Exception as error:

            print(f"Question processing error: {error}")

            chat_history.append({
                "role": "assistant",
                "content": "Sorry, I couldn't process your question. Please try again.",
                "sources": []
            })

        st.rerun()

else:
    st.info("Upload and process at least one PDF to start chatting.")


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        <h4>DocuMind AI</h4>
        <p>General-Purpose PDF Question Answering System</p>
        <p>Python | Streamlit | Google Gemini | LangChain | FAISS</p>
    </div>
    """,
    unsafe_allow_html=True
)
