"""Tests unitaires pour les steps du pipeline RAG.

Ces tests vous sont fournis. Votre code doit les passer.
Lancez : pytest tests/test_steps.py -v
"""

import tempfile
import numpy as np
import pytest


# ============================================================
# Tests — chunk_documents
# ============================================================
class TestChunkDocuments:

    def test_output_format(self):
        """Chaque chunk a les bonnes clés."""
        from steps.chunk_documents import chunk_documents

        docs = [("Texte long. " * 200, "test.pdf")]
        result = chunk_documents(
            documents=docs, chunk_size=100, chunk_overlap=20,
        )
        assert len(result) > 1
        for chunk in result:
            assert "content" in chunk
            assert "source" in chunk
            assert "chunk_id" in chunk
            assert chunk["source"] == "test.pdf"
            assert len(chunk["content"]) <= 120  # marge splitter

    def test_overlap(self):
        """L'overlap fonctionne : le début du chunk N+1
        contient la fin du chunk N."""
        from steps.chunk_documents import chunk_documents

        docs = [("Mot " * 500, "test.pdf")]
        result = chunk_documents(
            documents=docs, chunk_size=100, chunk_overlap=30,
        )
        if len(result) >= 2:
            fin = result[0]["content"][-30:]
            debut = result[1]["content"][:30]
            common = set(fin.split()) & set(debut.split())
            assert len(common) > 0

    def test_unique_ids(self):
        """Tous les chunk_id sont uniques."""
        from steps.chunk_documents import chunk_documents

        docs = [
            ("Texte. " * 200, "a.pdf"),
            ("Autre. " * 200, "b.pdf"),
        ]
        result = chunk_documents(
            documents=docs, chunk_size=50, chunk_overlap=10,
        )
        ids = [c["chunk_id"] for c in result]
        assert len(ids) == len(set(ids))

    def test_multiple_pdfs(self):
        """Les chunks de différents PDFs sont mélangés."""
        from steps.chunk_documents import chunk_documents

        docs = [
            ("Premier document. " * 100, "a.pdf"),
            ("Second document. " * 100, "b.pdf"),
        ]
        result = chunk_documents(
            documents=docs, chunk_size=80, chunk_overlap=10,
        )
        sources = set(c["source"] for c in result)
        assert sources == {"a.pdf", "b.pdf"}


# ============================================================
# Tests — generate_embeddings
# ============================================================
class TestGenerateEmbeddings:

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"content": "LoRA réduit les paramètres entraînables",
             "source": "t.pdf", "chunk_id": "t_0000"},
            {"content": "QLoRA quantifie les poids en 4 bits",
             "source": "t.pdf", "chunk_id": "t_0001"},
        ]

    def test_embeddings_added(self, sample_chunks):
        """Chaque chunk contient un embedding."""
        from steps.generate_embeddings import generate_embeddings

        result = generate_embeddings(chunks=sample_chunks)
        assert len(result) == 2
        for r in result:
            assert "embedding" in r
            assert isinstance(r["embedding"], list)
            assert len(r["embedding"]) == 384

    def test_embeddings_normalized(self):
        """Les embeddings sont normalisés (norme ~1)."""
        from steps.generate_embeddings import generate_embeddings

        chunks = [{"content": "Test embedding normalisation",
                   "source": "t.pdf", "chunk_id": "t_0000"}]
        result = generate_embeddings(chunks=chunks)
        vec = np.array(result[0]["embedding"])
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_no_mutation(self):
        """Les dicts d'entrée ne sont pas modifiés."""
        from steps.generate_embeddings import generate_embeddings

        chunks = [{"content": "Test", "source": "t.pdf",
                   "chunk_id": "t_0000"}]
        original_keys = set(chunks[0].keys())
        generate_embeddings(chunks=chunks)
        assert set(chunks[0].keys()) == original_keys

    def test_original_keys_preserved(self, sample_chunks):
        """Les clés originales sont préservées dans la sortie."""
        from steps.generate_embeddings import generate_embeddings

        result = generate_embeddings(chunks=sample_chunks)
        for r in result:
            assert "content" in r
            assert "source" in r
            assert "chunk_id" in r


# ============================================================
# Tests — index_documents
# ============================================================
class TestIndexDocuments:

    def test_creates_collection(self):
        """La collection est créée et contient les bons vecteurs."""
        import chromadb
        from steps.index_documents import index_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = [
                {"content": "Texte A", "source": "a.pdf",
                 "chunk_id": "a_0000", "embedding": [0.1] * 384},
                {"content": "Texte B", "source": "b.pdf",
                 "chunk_id": "b_0000", "embedding": [0.2] * 384},
            ]
            name = index_documents(
                chunks_with_embeddings=chunks,
                collection_name="test_col",
                persist_dir=tmpdir,
            )
            assert name == "test_col"
            client = chromadb.PersistentClient(path=tmpdir)
            col = client.get_collection("test_col")
            assert col.count() == 2

    def test_idempotent(self):
        """Relancer l'indexation repart de zéro (pas de doublons)."""
        import chromadb
        from steps.index_documents import index_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = [
                {"content": "X", "source": "x.pdf",
                 "chunk_id": "x_0", "embedding": [0.5] * 384},
            ]
            index_documents(
                chunks_with_embeddings=chunks,
                collection_name="test_idem",
                persist_dir=tmpdir,
            )
            index_documents(
                chunks_with_embeddings=chunks,
                collection_name="test_idem",
                persist_dir=tmpdir,
            )
            client = chromadb.PersistentClient(path=tmpdir)
            col = client.get_collection("test_idem")
            assert col.count() == 1

    def test_metadata_stored(self):
        """Les métadonnées de source sont stockées."""
        import chromadb
        from steps.index_documents import index_documents

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = [
                {"content": "Hello", "source": "my.pdf",
                 "chunk_id": "m_0", "embedding": [0.3] * 384},
            ]
            index_documents(
                chunks_with_embeddings=chunks,
                collection_name="test_meta",
                persist_dir=tmpdir,
            )
            client = chromadb.PersistentClient(path=tmpdir)
            col = client.get_collection("test_meta")
            data = col.get(include=["metadatas"])
            assert data["metadatas"][0]["source"] == "my.pdf"
