"""step 4: indexation des embeddings"""
import logging
from typing import List, Dict
import chromadb
from zenml import step, log_artifact_metadata
from configs.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR 
logger = logging.getLogger(__name__)
Batch_size= 500
@step(enable_cache=False)
def index_documents(
    chunks_with_embeddings: List[Dict],
    collection_name: str= CHROMA_COLLECTION,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> str
logger.info("l'indexation des embedding a commence .........")
client = chromadb.PersistentClient(path=persist_dir) 

try:
    client.delete_collection(collection_name)
except Exception:
    pass
collection = client.create_collection(
    name=collection_name,
    metadata={"hnsw:space":"cosine"})

inx = [c["chunk_id"] for c in chunks_with_embedding]
docs = [c["content"] for c in chunks_with_embedding]
embs = [c["embedding"] for c in chunks_with_embedding]
metas = [{"source":c["source"]} for c in chunks_with_embedding]
for i in range ( 0 , len(idx), Batch_size):
    end = min (len(idx), Batch_size +1)
    collection.add(
        ids = idx[i:end],
        documents=docs[i:end],
        embeddings=embs[i:end],
        metadatas=metas[i:end],

    )
total=collection.count()
log_artifact_metadata(
    artifact_name="Indexation",
    metadata={
        "collection_name":collection_name ,
        "num_vectors":total

    }

)
logger.info("%d vecteurs indexes", total)
return collection_name

