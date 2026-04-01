"""Tests unitaires pour les fonctions de retrieval.

Ces tests vous sont fournis. Votre code doit les passer.
Lancez : pytest tests/test_retrieval.py -v
"""

import tempfile
import pytest
import chromadb
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="module")
def indexed_collection():
    """Crée une collection ChromaDB temporaire avec des données de test."""
    tmpdir = tempfile.mkdtemp()
    client = chromadb.PersistentClient(path=tmpdir)

    try:
        client.delete_collection("test_retrieval")
    except Exception:
        pass

    col = client.create_collection(
        name="test_retrieval",
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Corpus volontairement diversifié pour que BM25 et Dense
    # ne retournent PAS le même classement sur toutes les requêtes
    docs = [
        ("LoRA utilise des matrices de rang faible pour adapter les LLMs. "
         "Le rank r contrôle la capacité du modèle. "
         "Les matrices A et B sont de dimensions d×r et r×k respectivement.",
         "lora.pdf"),
        ("QLoRA combine la quantification en 4 bits avec LoRA pour réduire "
         "la mémoire nécessaire au fine-tuning. QLoRA introduit le type "
         "NormalFloat4 et la double quantification.",
         "qlora.pdf"),
        ("Le Prefix-Tuning ajoute des vecteurs apprenables devant les "
         "séquences d'entrée au lieu de modifier les poids du modèle. "
         "Cette méthode est particulièrement efficace pour la génération.",
         "prefix.pdf"),
        ("BERT est un modèle de langage bidirectionnel pré-entraîné "
         "sur un large corpus de texte anglais. Il utilise le masked "
         "language modeling et le next sentence prediction.",
         "bert.pdf"),
        ("Le fine-tuning classique met à jour tous les paramètres du modèle, "
         "ce qui nécessite beaucoup de mémoire GPU et de temps de calcul. "
         "Les méthodes PEFT comme LoRA réduisent ce coût.",
         "ft.pdf"),
        ("L'optimiseur AdamW est couramment utilisé pour le fine-tuning "
         "des transformers. Il combine Adam avec une régularisation L2 "
         "découplée du taux d'apprentissage.",
         "adamw.pdf"),
        ("La quantification réduit la précision des poids du modèle de "
         "float32 à int8 ou int4, permettant de charger des modèles plus "
         "grands en mémoire GPU limitée.",
         "quantization.pdf"),
        ("Les adaptateurs sont de petits modules ajoutés entre les couches "
         "du transformer. Contrairement à LoRA qui modifie les poids via "
         "des matrices low-rank, les adaptateurs ajoutent de nouvelles couches.",
         "adapters.pdf"),
    ]

    texts = [d[0] for d in docs]
    embeddings = model.encode(texts, normalize_embeddings=True)

    col.add(
        ids=[f"doc_{i}" for i in range(len(docs))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": d[1]} for d in docs],
    )

    yield {
        "collection_name": "test_retrieval",
        "persist_dir": tmpdir,
    }


class TestDenseRetrieve:

    def test_format(self, indexed_collection):
        """dense_retrieve retourne le bon format."""
        from utils.retrieval import dense_retrieve

        results = dense_retrieve(
            "test query", top_k=3, **indexed_collection,
        )
        assert len(results) <= 3
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "distance" in r

    def test_top_k_respected(self, indexed_collection):
        """Le nombre de résultats respecte top_k."""
        from utils.retrieval import dense_retrieve

        results = dense_retrieve(
            "LoRA rank", top_k=2, **indexed_collection,
        )
        assert len(results) == 2

    def test_relevance(self, indexed_collection):
        """Une question sur LoRA retourne le doc LoRA en premier."""
        from utils.retrieval import dense_retrieve

        results = dense_retrieve(
            "Comment fonctionne LoRA ?", top_k=3,
            **indexed_collection,
        )
        assert len(results) > 0
        assert "lora" in results[0]["source"].lower() or \
               "lora" in results[0]["content"].lower()


class TestHybridRetrieve:

    def test_format(self, indexed_collection):
        """hybrid_retrieve retourne le bon format."""
        from utils.retrieval import hybrid_retrieve

        results = hybrid_retrieve(
            "test query", top_k=3, bm25_weight=0.4,
            **indexed_collection,
        )
        assert len(results) <= 3
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "score" in r

    def test_scores_positive(self, indexed_collection):
        """Les scores sont positifs."""
        from utils.retrieval import hybrid_retrieve

        results = hybrid_retrieve(
            "LoRA", top_k=3, bm25_weight=0.5,
            **indexed_collection,
        )
        for r in results:
            assert r["score"] > 0

    def test_weights_affect_ranking(self, indexed_collection):
        """Changer bm25_weight change le classement ou les scores.

        On utilise une requête avec un terme technique exact
        ('NormalFloat4') qui favorise BM25, et un sens sémantique
        ('réduire la mémoire') qui favorise Dense.
        """
        from utils.retrieval import hybrid_retrieve

        query = "NormalFloat4 réduire mémoire fine-tuning"

        r1 = hybrid_retrieve(
            query, bm25_weight=0.95,
            top_k=4, **indexed_collection,
        )
        r2 = hybrid_retrieve(
            query, bm25_weight=0.05,
            top_k=4, **indexed_collection,
        )

        if len(r1) > 0 and len(r2) > 0:
            contents1 = [r["content"][:50] for r in r1]
            contents2 = [r["content"][:50] for r in r2]
            scores1 = [round(r["score"], 6) for r in r1]
            scores2 = [round(r["score"], 6) for r in r2]
            assert contents1 != contents2 or scores1 != scores2, (
                "Les résultats sont identiques malgré des poids "
                "très différents (0.95 vs 0.05)."
            )

    def test_top_k_respected(self, indexed_collection):
        """Le nombre de résultats respecte top_k."""
        from utils.retrieval import hybrid_retrieve

        results = hybrid_retrieve(
            "LoRA", top_k=2, bm25_weight=0.4,
            **indexed_collection,
        )
        assert len(results) <= 2
