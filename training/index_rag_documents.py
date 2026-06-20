import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.rag_service import RAGService


def load_documents(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("RAG document input must be a JSON list")
    return payload


async def index_documents(path: Path) -> None:
    docs = load_documents(path)
    RAGService.initialize()
    if not RAGService.initialized:
        raise RuntimeError("RAG service failed to initialize")
    await RAGService.index_documents(docs)
    print(f"Indexed {len(docs)} source documents into ChromaDB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index JSON documents into ChromaDB for RAG.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("training/examples/realtime_rag_documents.json"),
        help="JSON list of documents with id, text, and metadata fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(index_documents(args.docs))


if __name__ == "__main__":
    main()
