#!/usr/bin/env python3
"""Point d'entrée principal du projet RAG ZenML.

Usage :
    python run.py index --pdf-dir ./pdfs
    python run.py query --question "..." [--hybrid]
    python run.py eval [--top-k 5]
    python run.py history [--limit 10]
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag-zenml")


def cmd_index(args: argparse.Namespace) -> None:
    """Exécute le pipeline d'indexation."""
    from pipelines import rag_indexing_pipeline

    logger.info("=" * 60)
    logger.info("PIPELINE D'INDEXATION RAG")
    logger.info(
        "  pdf_dir=%s  chunk_size=%d  chunk_overlap=%d",
        args.pdf_dir,
        args.chunk_size,
        args.chunk_overlap,
    )
    logger.info("=" * 60)

    rag_indexing_pipeline(
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    logger.info("Pipeline d'indexation terminé.")


def cmd_query(args: argparse.Namespace) -> None:
    """Exécute le pipeline d'inférence."""
    from zenml.client import Client

    if args.hybrid:
        from pipelines import rag_hybrid_inference_pipeline

        pipeline_name = "rag_hybrid_inference_pipeline"
        step_name = "rag_hybrid_inference"

        logger.info("=" * 60)
        logger.info("INFÉRENCE RAG HYBRIDE (BM25 + Dense)")
        logger.info("  question : %s", args.question)
        logger.info("  bm25_weight : %.2f", args.bm25_weight)
        logger.info("=" * 60)

        rag_hybrid_inference_pipeline(
            question=args.question,
            bm25_weight=args.bm25_weight,
        )
    else:
        from pipelines import rag_dense_inference_pipeline

        pipeline_name = "rag_dense_inference_pipeline"
        step_name = "rag_inference"

        logger.info("=" * 60)
        logger.info("INFÉRENCE RAG DENSE")
        logger.info("  question : %s", args.question)
        logger.info("=" * 60)

        rag_dense_inference_pipeline(question=args.question)

    # Récupérer et afficher le résultat
    client = Client()
    last_run = client.get_pipeline(pipeline_name).last_run
    output = last_run.steps[step_name].output.load()

    print("\n" + "=" * 60)
    print(f"Question : {output['question']}")
    print("=" * 60)
    print(f"\nRéponse :\n{output['answer']}")
    print(f"\nSources  : {output['sources']}")
    print(f"Retriever : {output['retriever']}")
    print(f"Chunks    : {output['num_chunks']}")
    print("=" * 60)


def cmd_eval(args: argparse.Namespace) -> None:
    """Exécute le pipeline d'évaluation RAGAS."""
    from zenml.client import Client

    from pipelines import rag_evaluation_pipeline

    logger.info("=" * 60)
    logger.info("ÉVALUATION RAGAS")
    logger.info("  top_k=%d", args.top_k)
    logger.info("=" * 60)

    rag_evaluation_pipeline(top_k=args.top_k)

    # Récupérer et afficher les résultats
    client = Client()
    last_run = client.get_pipeline("rag_evaluation_pipeline").last_run
    output = last_run.steps["evaluate_rag"].output.load()

    print("\n" + "=" * 60)
    print("RÉSULTATS RAGAS")
    print("=" * 60)
    print(f"  Faithfulness      : {output['avg_faithfulness']:.3f}")
    print(f"  Answer Relevancy  : {output['avg_answer_relevancy']:.3f}")
    print(f"  Context Precision : {output['avg_context_precision']:.3f}")
    print(f"  Questions évaluées : {output['num_questions']}")

    if output.get("low_faithfulness_questions"):
        print("\n  Questions avec Faithfulness < 0.8 :")
        for q in output["low_faithfulness_questions"]:
            print(f"    ⚠  {q}")

    if output.get("low_precision_questions"):
        print("\n  Questions avec Context Precision < 0.7 :")
        for q in output["low_precision_questions"]:
            print(f"    ⚠  {q}")

    print("\n  Détail par question :")
    for pq in output.get("per_question", []):
        print(f"    Q: {pq['question'][:50]}...")
        print(
            f"       faith={pq['faithfulness']:.3f}  "
            f"relev={pq['answer_relevancy']:.3f}  "
            f"prec={pq['context_precision']:.3f}"
        )

    print("=" * 60)


def cmd_history(args: argparse.Namespace) -> None:
    """Affiche l'historique des runs ZenML."""
    from zenml.client import Client

    client = Client()

    for pipeline_name in [
        "rag_indexing_pipeline",
        "rag_dense_inference_pipeline",
        "rag_hybrid_inference_pipeline",
        "rag_evaluation_pipeline",
    ]:
        try:
            pipeline_model = client.get_pipeline(pipeline_name)
        except Exception:
            continue

        runs = pipeline_model.runs
        if not runs:
            continue

        print(f"\n{'=' * 60}")
        print(f"Pipeline : {pipeline_name}")
        print(f"{'=' * 60}")

        for run in runs[: args.limit]:
            print(f"\n  {run.name}")
            print(f"     Status : {run.status}")
            print(f"     Date   : {run.created}")

            for step_name, step_info in run.steps.items():
                duration = step_info.duration or "N/A"
                print(f"     └─ {step_name} ({step_info.status}, {duration})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline RAG avec ZenML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commande")

    # ── index ─────────────────────────────────────────────
    p_index = subparsers.add_parser("index", help="Indexer les PDFs")
    p_index.add_argument("--pdf-dir", type=str, default="./pdfs")
    p_index.add_argument("--chunk-size", type=int, default=800)
    p_index.add_argument("--chunk-overlap", type=int, default=100)

    # ── query ─────────────────────────────────────────────
    p_query = subparsers.add_parser("query", help="Poser une question")
    p_query.add_argument("--question", type=str, required=True)
    p_query.add_argument("--hybrid", action="store_true")
    p_query.add_argument("--bm25-weight", type=float, default=0.4)

    # ── eval ──────────────────────────────────────────────
    p_eval = subparsers.add_parser("eval", help="Évaluation RAGAS")
    p_eval.add_argument("--top-k", type=int, default=3)

    # ── history ───────────────────────────────────────────
    p_history = subparsers.add_parser("history", help="Historique des runs")
    p_history.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "index": cmd_index,
        "query": cmd_query,
        "eval": cmd_eval,
        "history": cmd_history,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
