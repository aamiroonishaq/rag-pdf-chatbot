"""
Gradio web interface for the RAG document Q&A system.
Upload a PDF, then ask questions about its content.
"""

import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

rag = None  # initialized after the user provides an API key (or env var is set)


def init_pipeline(api_key: str):
    global rag
    try:
        rag = RAGPipeline(google_api_key=api_key or None)
        return "✅ Pipeline initialized. Now upload a PDF."
    except ValueError as e:
        return f"❌ {e}"


def upload_and_index(file):
    global rag
    if rag is None:
        return "⚠️ Please initialize the pipeline with an API key first."
    if file is None:
        return "⚠️ No file uploaded."

    dest_path = os.path.join(UPLOAD_DIR, os.path.basename(file.name))
    shutil.copy(file.name, dest_path)

    num_chunks = rag.load_and_index(dest_path)
    return f"✅ Indexed '{os.path.basename(file.name)}' into {num_chunks} chunks. You can now ask questions."


def answer_question(question, history):
    global rag
    if rag is None or rag.chain is None:
        return "⚠️ Please initialize the pipeline and upload a document first."
    return rag.ask(question)


with gr.Blocks(title="RAG Document Q&A") as demo:
    gr.Markdown("# 📄 RAG-based Document Q&A\nUpload a PDF and ask questions about its content using LangChain + Chroma + Google Gemini (free tier).")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Google API Key",
            type="password",
            placeholder="AIza...",
            scale=4,
        )
        init_btn = gr.Button("Initialize", scale=1)
    init_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Index Document")
    upload_status = gr.Textbox(label="Indexing Status", interactive=False)

    gr.Markdown("---")
    chatbot = gr.ChatInterface(
        fn=answer_question,
        title="Ask Questions About Your Document",
    )

    init_btn.click(init_pipeline, inputs=api_key_input, outputs=init_status)
    upload_btn.click(upload_and_index, inputs=file_input, outputs=upload_status)


if __name__ == "__main__":
    demo.launch()