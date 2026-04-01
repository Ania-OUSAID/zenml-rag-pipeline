""" step 3: generation des embiding """
import logging
from typing import Annotated, Dict, List
from zenml import step, log_artifact_metadata
from sentence_transformers import SentenceTransformer
from configs.settings import EMBEDDING_MODEL
logger= logging.getLogger(__name__)
@step(enable_cache= True)
def generated_embeddings(
    chunks: List[Dict[str, str]],
    model_name: str = EMBEDDING_MODEL
) -> Annotated[list[Dict], "chunks_with_embeddings"]:
    logger.info("chargement du model %s", model_name)

model = SentenceTransformer(model_name)
texts=[c["content"] for  c in chunks]
logger.info("encodage de %d chunks....", len(texts))
embiddings= model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True,
    batch_size=64,

)
enriched: List[Dict]=[]
for chunk, emb in zip(chunks,embiddings):
    enriched_chunk= chunk.copy
    enriched_chunk["embeddings"] = emb.tolist()
    enriched.append(enriched_chunk)
    log_artifact_metadata(
        artifact_name="chnuks_with_embeddings",
        metadata={
            "model_name": model_name,
            "embedding_dim":len(embiddings[0]),
            "num_embedding": len(embiddings)
        },
    )
    logger.info("%d embeddings generes", len(embiddings))
    return enriched