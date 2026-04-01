"""
step 1:chargement des pdfs
pour charger les docs pdf de note repertoire
"""
import logging
import os   
import glob
from typing import List, Tuple , Annotated
from zenml import log_artifact_metadata, step
import fitz
logger = logging.getLogger(__name__)
@step(enable_cache=True)
def load_pdfs(
    pdf_dir:str,
) -> Annotated[list[Tuple[str, str]],"raw_documents"]
"""
obj : charger les documents pdfs d un dossier
args:
pdf_dir ; chemin vers le dossier contenant des pdfs
returns: liste de tuples ( contenu_teste, nom_fichiers)
"""
pdf_files= sorted(glob.glob(os.path.join(pdf_dir,"*.pdf")))
if not pdf_files:
    raise ValueError(f"aucon pdf trouve {pdf_files}")
documents: List[Tuple[str, str]]=[]
total_pages = 0
for pdf_path in pdf_files:
    filename=os.path.basename(pdf_path)
    try:
        doc=fitz.open(pdf_dir)
        pages=[page.get_text() for page in doc]
        text="\n".join(pages)
        num_pages= len(doc)
        doc.close()
        documents.append((text, filename))
        num_pages += len(doc)
        doc.close()
        logger.info(
            "chargre %s - %d  pages , %d  caracteres",
            filename, num_pages, len(text)
        )
    except Exception as e:
        logger.error("erreur lors du chergement de %s:%s", filename , e)
        raise
log_artifact_metadata(
    artifact_name="raw_documents",
    metadata={
        "num_pdf": len(documents),
        "total_pages":total_pages,
        "total_characters": sum(len(d[0]) for d in documents)
        "pdf_names": [d[1] for d in documents]
    }
)
