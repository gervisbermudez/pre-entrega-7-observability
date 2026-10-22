"""RAG híbrido de las políticas de TechCorp.

Reconecta el índice ``techcorp-rag-hibrido`` (pre-entrega 4). Si el namespace
ya tiene vectores, no reingesta. Si Pinecone no responde, queda BM25 sobre
``data/*.txt``, que alcanza para la búsqueda del investigador.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_NAME_DEFAULT = "techcorp-rag-hibrido"
NAMESPACE = "politicas-internas"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5


class RAGSystem:
    def __init__(self, retriever: BaseRetriever, k: int = TOP_K):
        self.retriever = retriever
        self.k = k

    def obtener_top_k(self, query: str) -> list[dict[str, str]]:
        docs = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "?"),
            }
            for d in docs
        ]


def cargar_chunks():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documentos = loader.load()
    if not documentos:
        raise FileNotFoundError(f"No hay archivos .txt en {DATA_DIR}")

    chunks = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=100,
    ).split_documents(documentos)
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = os.path.basename(chunk.metadata["source"])
        chunk.metadata["chunk_id"] = i
    return chunks


def _retriever_pinecone(chunks):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key or api_key.startswith("tu_clave"):
        raise RuntimeError("Falta PINECONE_API_KEY")

    index_name = os.getenv("INDEX_NAME", INDEX_NAME_DEFAULT).strip() or INDEX_NAME_DEFAULT
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    pc = Pinecone(api_key=api_key)
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        raise RuntimeError(
            f"No existe el índice {index_name}. "
            "Crealo con la pre-entrega 4 antes de correr el orquestador."
        )

    dimension = int(pc.describe_index(index_name).dimension)
    if dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"El índice {index_name} tiene dimensión {dimension} y "
            f"{EMBEDDING_MODEL} usa {EMBEDDING_DIM}."
        )

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=NAMESPACE,
    )
    stats = pc.Index(index_name).describe_index_stats()
    namespaces = getattr(stats, "namespaces", None) or {}
    info = namespaces.get(NAMESPACE)
    vector_count = 0
    if info is not None:
        vector_count = int(getattr(info, "vector_count", 0) or 0)
    if vector_count == 0:
        print("Namespace vacío — ingestando chunks...")
        vectorstore.add_documents(chunks)
    else:
        print("Namespace ya tiene datos — no se reingesta")

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K, "namespace": NAMESPACE})


_rag: RAGSystem | None = None
_rag_lock = threading.Lock()


def obtener_rag() -> RAGSystem:
    """Construye el retriever una sola vez, aunque las búsquedas salgan en paralelo."""
    global _rag
    if _rag is not None:
        return _rag

    with _rag_lock:
        if _rag is not None:
            return _rag

        chunks = cargar_chunks()
        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = TOP_K

        try:
            vectorial = _retriever_pinecone(chunks)
            retriever: BaseRetriever = EnsembleRetriever(
                retrievers=[bm25, vectorial],
                weights=[0.5, 0.5],
            )
            print("RAGSystem reconectado (BM25 + Pinecone)")
        except Exception as exc:
            print(f"Pinecone no disponible ({exc}). Se usa BM25 sobre data/.")
            retriever = bm25

        _rag = RAGSystem(retriever)
        return _rag
