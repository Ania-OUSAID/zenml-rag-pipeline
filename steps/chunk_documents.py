"""step 2: decouper les document en chunks"""
import logging
from typing import Annotated, List, Dict, Tuple
logger = logging.getLogger(__name__)
from configs.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from zenml import log_artifact_metadata, step
from langchain_text_splitters import RecursiveCharacterTextSplitter
@step(enable_cache=True)
def chunk_documents(
    documents: list(Tuple[str, str]),
    chunk_size: int= DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
)-> Annotated[list[Dict[str, str]] , "chunk"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap =chunk_overlap,
        len_function  =len,
        separators= ["\n\n", "\n","."," ",""]  

    )
    all_chunks: List[dict[str, str]] = []
    for text, filename in documents:
        text_chunks = splitter.split_text(text)
        for i , chunk_text in enumerate(text_chunks):
            all_chunks.append({
                "content": chunk_text,
                "source" : filename,
                "chunk_id": f"{filename}_chunk_{i:04d}"
            })
    lengths= [len(c["content"]) for c in all_chunks]
    log_artifact_metadata(
        artifact_name="chunk",
        metadata={
            "chuns_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "total_chunk": len(all_chunks),
            "avg_chunk_lenght": sum(lengths) // max(len(lengths),1)
        }
    )
    logger.info("%d chuks generes", len (all_chunks))
    return all_chunks