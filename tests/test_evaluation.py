"""Tests pour le step d'évaluation RAGAS.

Lancez : pytest tests/test_evaluation.py -v
Pour les tests d'intégration : pytest tests/test_evaluation.py -v -m slow
"""

import tempfile
import pytest
import chromadb
from sentence_transformers import SentenceTransformer


# ── Fixture pour les tests d'intégration ─────────────────
@pytest.fixture(scope="module")
def indexed_collection_with_data():
    """Crée une collection ChromaDB avec des données pour
    tester l'évaluation RAGAS de bout en bout."""
    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)

    try:
        client.delete_collection("test_eval")
    except Exception:
        pass

    col = client.create_collection(
        name="test_eval",
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    docs = [
        "LoRA utilise des matrices de rang faible A et B "
        "pour adapter les LLMs sans modifier tous les poids. "
        "Le rank r contrôle le nombre de paramètres ajoutés.",
        "QLoRA combine la quantification en 4 bits avec LoRA. "
        "Cela réduit la mémoire GPU nécessaire au fine-tuning "
        "tout en maintenant les performances.",
        "Le Prefix-Tuning ajoute des vecteurs apprenables "
        "devant les séquences d'entrée, permettant d'adapter "
        "le modèle sans modifier ses poids internes.",
    ]

    embeddings = model.encode(
        docs, normalize_embeddings=True
    )

    col.add(
        ids=[f"eval_{i}" for i in range(len(docs))],
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": f"doc{i}.pdf"} for i in range(len(docs))],
    )

    yield {
        "collection_name": "test_eval",
        "persist_dir": tmpdir,
    }


class TestEvaluateRagOutputFormat:
    """Tests sur le format de sortie (sans appel LLM)."""

    def test_default_questions_exist(self):
        """Les questions d'évaluation par défaut sont définies."""
        from steps.evaluate_rag import DEFAULT_EVAL_QUESTIONS

        assert isinstance(DEFAULT_EVAL_QUESTIONS, list)
        assert len(DEFAULT_EVAL_QUESTIONS) >= 3
        for q in DEFAULT_EVAL_QUESTIONS:
            assert isinstance(q, str)
            assert len(q) > 10

    def test_step_is_decorated(self):
        """evaluate_rag est bien un step ZenML."""
        from steps.evaluate_rag import evaluate_rag

        assert hasattr(evaluate_rag, "entrypoint")

    def test_output_keys_documented(self):
        """Vérifie que le docstring mentionne les métriques."""
        from steps.evaluate_rag import evaluate_rag

        doc = evaluate_rag.entrypoint.__doc__ or ""
        assert "faithfulness" in doc.lower() or \
               "Faithfulness" in doc


class TestEvaluateRagIntegration:
    """Tests d'intégration (nécessitent collection + API key).

    Marqués 'slow' — skippés par défaut.
    Lancez avec : pytest -m slow
    """

    @pytest.mark.slow
    def test_full_evaluation(self, indexed_collection_with_data):
        """Test complet du step d'évaluation."""
        from steps.evaluate_rag import evaluate_rag

        result = evaluate_rag(
            eval_questions=[
                "Comment fonctionne LoRA ?",
                "Qu'est-ce que QLoRA ?",
            ],
            top_k=3,
            **indexed_collection_with_data,
        )

        assert "avg_faithfulness" in result
        assert "avg_answer_relevancy" in result
        assert "avg_context_precision" in result
        assert "num_questions" in result
        assert "per_question" in result

        assert isinstance(result["avg_faithfulness"], float)
        assert result["num_questions"] == 2
        assert 0 <= result["avg_faithfulness"] <= 1
        assert 0 <= result["avg_answer_relevancy"] <= 1
        assert 0 <= result["avg_context_precision"] <= 1

        assert len(result["per_question"]) == 2
        for pq in result["per_question"]:
            assert "question" in pq
            assert "faithfulness" in pq
            assert "answer_relevancy" in pq
            assert "context_precision" in pq
