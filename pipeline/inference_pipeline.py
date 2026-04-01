"""Pipelines d'inférence RAG."""

from typing import Dict
from zenml import pipeline
from steps import rag_inference, rag_hybrid_inference


@pipeline(name="rag_dense_inference_pipeline")
def rag_dense_inference_pipeline(question: str) -> Dict[str, str]:
    return rag_inference(question=question)


@pipeline(name="rag_hybrid_inference_pipeline")
def rag_hybrid_inference_pipeline(
    question: str, bm25_weight: float = 0.4,
) -> Dict[str, str]:
    return rag_hybrid_inference(question=question, bm25_weight=bm25_weight)
