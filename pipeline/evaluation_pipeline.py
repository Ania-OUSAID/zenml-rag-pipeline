"""Pipeline d'évaluation RAG avec RAGAS."""

from typing import Dict, List
from zenml import pipeline
from configs.settings import DEFAULT_TOP_K
from steps import evaluate_rag
from steps.evaluate_rag import DEFAULT_EVAL_QUESTIONS


@pipeline(name="rag_evaluation_pipeline")
def rag_evaluation_pipeline(
    eval_questions: List[str] = DEFAULT_EVAL_QUESTIONS,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, object]:
    return evaluate_rag(eval_questions=eval_questions, top_k=top_k)
