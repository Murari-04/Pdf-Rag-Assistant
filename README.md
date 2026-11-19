# PDF RAG Assistant

An AI-powered PDF question-answering application that allows users to upload PDF documents and ask questions in natural language. It uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate context-based answers using Google Gemini.

## Overview

PDF RAG Assistant makes it easier to understand and interact with PDF documents. Instead of manually searching through lengthy documents, users can upload their PDFs and ask questions to get relevant answers with source page references.

The application uses embeddings and a FAISS vector database to retrieve relevant document content before generating answers with Gemini.

## Features

- **Multiple PDF Uploads:** Upload and work with multiple PDF documents.
- **AI-Powered Question Answering:** Ask questions in natural language.
- **Retrieval-Augmented Generation:** Retrieve relevant document content before generating answers.
- **Semantic Search:** Find relevant information using vector embeddings.
- **Source References:** View source document names and page numbers used to generate answers.
- **Document Selection:** Select individual documents to ask questions about.
- **Chat History:** Maintain conversation history for each selected document.
- **Suggested Questions:** Get question suggestions to explore document content.
- **Session-Based Storage:** Store vector databases in separate browser-session directories.
- **File Validation:** Validate PDF uploads and enforce file size and upload count limits.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | RAG workflow and document processing |
| Google Gemini 2.5 Flash | AI answer generation |
| Gemini Embedding 001 | Text embedding generation |
| FAISS | Vector database and similarity search |
| PyPDF | PDF text extraction |
| python-dotenv | Environment variable management |

## Project Architecture

The application follows a Retrieval-Augmented Generation workflow:

1. **PDF Upload:** Users upload PDF documents through the Streamlit interface.
2. **Text Extraction:** PyPDF extracts text from uploaded documents.
3. **Text Chunking:** Extracted text is divided into smaller chunks.
4. **Embedding Generation:** Gemini converts text chunks into numerical vector embeddings.
5. **Vector Storage:** FAISS stores the embeddings for efficient similarity search.
6. **Question Processing:** The user's question is converted into an embedding and matched against stored document vectors.
7. **Context Retrieval:** Relevant document chunks are retrieved.
8. **Answer Generation:** Gemini generates an answer using the retrieved context.
9. **Source Display:** The application displays relevant source documents and page references.

## Project Structure

```text
PDF-RAG-Assistant/
│
├── app.py
├── config.py
├── pdf_processor.py
├── text_chunker.py
├── embedding_generator.py
├── vector_store.py
├── retriever.py
├── rag.py
├── llm_generator.py
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

**Note:** The `vectorstore/`, `.env`, and virtual environment folders are generated or maintained locally and should not be committed to GitHub.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Murari-04/Pdf-Rag-Assistant.git
```

Navigate to the project directory:

```bash
cd Pdf-Rag-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file in the project root directory.

Add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
```

You can obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Do not upload your `.env` file or expose your API key publicly.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Launch the application.
2. Upload one or more PDF documents.
3. Wait for the documents to be processed and indexed.
4. Select the document you want to interact with.
5. Enter a question in the chat interface.
6. Receive an AI-generated answer based on the document content.
7. Check the source references and page numbers to verify the answer.

## Configuration

The application uses the following default settings:

| Setting | Value |
|---|---|
| Embedding Model | gemini-embedding-001 |
| LLM Model | gemini-2.5-flash |
| Chunk Size | 1000 |
| Chunk Overlap | 200 |
| Retrieval Top K | 4 |
| Maximum PDF Size | 20 MB |
| Maximum Upload Count | 5 |

These settings can be adjusted in the application configuration.

## Important Notes

- An active Google Gemini API key is required.
- API usage is subject to Google's rate limits and quotas.
- Answers depend on the information available in the uploaded documents.
- The application may not answer questions when relevant information is unavailable in the selected documents.
- Source references help users verify the retrieved information.
- Session-based storage separates vector database directories between browser sessions; it should not be considered a complete authentication or security system.

## Future Enhancements

- Support for additional document formats.
- Improved document management.
- Streaming AI responses.
- Deployment to a cloud platform.
- Enhanced retrieval and response evaluation.

## Author

**Murari**

GitHub: [Murari-04](https://github.com/Murari-04)

---

⭐ If you find this project useful, consider giving the repository a star!
