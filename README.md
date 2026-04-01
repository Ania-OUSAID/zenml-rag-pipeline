# TP : Pipeline RAG avec ZenML

Pipeline RAG (Retrieval-Augmented Generation) structuré en projet Python
production avec ZenML pour l'orchestration, le versioning des artefacts
et la reproductibilité.

## Architecture du projet

```
rag-zenml-tp/
├── configs/
│   └── settings.py          # Configuration centralisée
├── steps/
│   ├── __init__.py
│   ├── load_pdfs.py          # Step 1 : Chargement des PDFs
│   ├── chunk_documents.py    # Step 2 : Découpage en chunks
│   ├── generate_embeddings.py# Step 3 : Vectorisation
│   ├── index_documents.py    # Step 4 : Indexation ChromaDB
│   └── rag_inference.py      # Step 5 : Inférence RAG
├── pipelines/
│   ├── __init__.py
│   ├── indexing_pipeline.py  # Pipeline d'indexation
│   └── inference_pipeline.py # Pipeline d'inférence
├── utils/
│   ├── __init__.py
│   └── retrieval.py          # Fonctions de retrieval (dense, hybrid)
├── run.py                    # Point d'entrée principal
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
zenml init
```

## Configuration

Créez un fichier `.env` ou exportez les variables :

```bash
export GROQ_API_KEY="votre_clé_groq"
```

## Utilisation

### Indexation (charger les PDFs et construire la base vectorielle)

```bash
python run.py index --pdf-dir ./pdfs --chunk-size 800 --chunk-overlap 100
```

### Inférence (poser une question)

```bash
python run.py query --question "Comment LoRA réduit-il le nombre de paramètres ?"
```

### Inférence hybride (BM25 + Dense)

```bash
python run.py query --question "Quel est le rank r dans LoRA ?" --hybrid
```

### Voir l'historique des runs

```bash
python run.py history
```

## Corpus attendu

Placez vos PDFs dans un dossier (par défaut `./pdfs/`) :
- LoRA, QLoRA, Prefix-Tuning, BERT, PEFT...

## Niveau : Master 2
## Durée estimée : ~2h
