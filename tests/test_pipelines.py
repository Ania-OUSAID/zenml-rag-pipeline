"""Tests pour les pipelines ZenML.

Ces tests vous sont fournis. Votre code doit les passer.
Lancez : pytest tests/test_pipelines.py -v
"""

import tempfile

import chromadb
import fitz
import pytest

from configs.settings import CHROMA_COLLECTION, CHROMA_PERSIST_DIR


@pytest.fixture(scope="module")
def tmp_pdf_dir():
    """Crée un dossier temporaire contenant un PDF minimal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Créer un PDF minimal avec du contenu sur LoRA
        pdf_path = f"{tmpdir}/test_lora.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            (
                "LoRA réduit le nombre de paramètres entraînables "
                "en injectant des matrices de rang inférieur dans chaque couche. "
                "Le rang r contrôle la capacité d'adaptation du modèle. "
                "Les matrices A et B sont initialisées aléatoirement et à zéro. "
                "QLoRA combine LoRA avec la quantification en 4 bits. "
                "Prefix-Tuning ajoute des vecteurs apprenables à l'entrée."
            ),
        )
        doc.save(pdf_path)
        doc.close()
        yield tmpdir


class TestIndexingPipeline:

    def test_pipeline_runs(self, tmp_pdf_dir):
        """Le pipeline s'exécute sans erreur et produit une collection."""
        from pipelines.indexing_pipeline import rag_indexing_pipeline

        rag_indexing_pipeline(
            pdf_dir=tmp_pdf_dir,
            chunk_size=200,
            chunk_overlap=20,
        )

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        col = client.get_collection(CHROMA_COLLECTION)
        assert col.count() > 0

    def test_pipeline_returns_collection_name(self, tmp_pdf_dir):
        """Le pipeline retourne le nom de la collection."""
        from zenml.client import Client

        client = Client()
        last_run = client.get_pipeline("rag_indexing_pipeline").last_run
        result = last_run.steps["index_documents"].output.load()
        assert result == CHROMA_COLLECTION

    def test_pipeline_is_idempotent(self, tmp_pdf_dir):
        """Relancer le pipeline ne crée pas de doublons."""
        from pipelines.indexing_pipeline import rag_indexing_pipeline

        rag_indexing_pipeline(
            pdf_dir=tmp_pdf_dir,
            chunk_size=200,
            chunk_overlap=20,
        )
        count_after_first = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR
        ).get_collection(CHROMA_COLLECTION).count()

        rag_indexing_pipeline(
            pdf_dir=tmp_pdf_dir,
            chunk_size=200,
            chunk_overlap=20,
        )
        count_after_second = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR
        ).get_collection(CHROMA_COLLECTION).count()

        assert count_after_first == count_after_second
