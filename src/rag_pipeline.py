"""
RAG Pipeline: handles document loading, chunking, embedding, vector storage,
and retrieval-augmented question answering using Google's Gemini models (free tier).
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "chroma_store"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the
provided context from a document. If the answer is not contained in the context,
say "I couldn't find that information in the document."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, google_api_key: str | None = None):
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY env variable "
                "or pass it directly. Get a free key at "
                "https://aistudio.google.com/app/apikey"
            )
        self.embeddings = GoogleGenerativeAIEmbeddings(
              model="models/gemini-embedding-001", google_api_key=api_key
       )
        self.llm = ChatGoogleGenerativeAI(
              model="gemini-2.5-flash", temperature=0, google_api_key=api_key
        )
        self.vectorstore: Chroma | None = None
        self.chain = None

    def load_and_index(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
        """Load a PDF, split it into chunks, and build a Chroma vector store."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR,
        )

        self._build_chain()
        return len(chunks)

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        if self.chain is None:
            return "Please upload and index a document first."
        return self.chain.invoke(question)