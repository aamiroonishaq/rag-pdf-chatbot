# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF
and ask natural-language questions about its content. Built with **LangChain**,
**Chroma** (vector database), and **OpenAI** models, with a **Gradio** web UI.

## How it works

1. **Load** – the uploaded PDF is parsed with `PyPDFLoader`.
2. **Chunk** – the document is split into overlapping chunks using
   `RecursiveCharacterTextSplitter` for better retrieval context.
3. **Embed & Store** – each chunk is embedded with `text-embedding-3-small`
   and stored in a local **Chroma** vector store.
4. **Retrieve & Generate** – on each question, the top-matching chunks are
   retrieved and passed as context to `gpt-4o-mini`, which generates an
   answer grounded in the document.

## Project Structure

```
rag-doc-qa/
├── app.py                 # Gradio UI entry point
├── src/
│   └── rag_pipeline.py    # Core RAG pipeline (load, chunk, embed, retrieve, answer)
├── data/                   # Uploaded PDFs are stored here
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd rag-doc-qa
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add your OpenAI API key:

```
OPENAI_API_KEY=your-openai-api-key-here
```

## Running the app

```bash
python app.py
```

This launches a local Gradio interface where you can:

1. Enter your OpenAI API key (or rely on the `.env` file) and click **Initialize**.
2. Upload a PDF and click **Index Document**.
3. Ask questions about the document in the chat interface.

## Tech Stack

- **LangChain** – orchestration of the RAG pipeline
- **Chroma** – vector database for similarity search
- **OpenAI** – embeddings (`text-embedding-3-small`) and chat model (`gpt-4o-mini`)
- **Gradio** – web interface
- **PyPDF** – PDF parsing

## Possible Extensions

- Support for multiple file formats (DOCX, TXT, web pages)
- Multi-document collections with source citations
- Local/open-source LLMs and embedding models (e.g., via Ollama)
- Conversation memory for multi-turn follow-up questions

## License

MIT
