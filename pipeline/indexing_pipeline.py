"""Pipeline d'indexation RAG."""

from zenml import pipeline
from configs.settings import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, PDF_DIR
from steps import chunk_documents, generate_embeddings, index_documents, load_pdfs


@pipeline(name="rag_indexing_pipeline", enable_cache=True)
def rag_indexing_pipeline(
    pdf_dir: str = PDF_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """Pipeline complet d'indexation RAG."""
    raw_docs = load_pdfs(pdf_dir=pdf_dir)
    chunks = chunk_documents(
        documents=raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    chunks_emb = generate_embeddings(chunks=chunks)
    return index_documents(chunks_with_embeddings=chunks_emb)
