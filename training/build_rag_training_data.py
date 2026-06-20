import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_records(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if isinstance(payload, list):
        return payload
    raise ValueError("input must be a raw collector JSON object or a JSON list")


def record_to_text(record: Dict[str, Any]) -> str:
    title = str(record.get("title") or record.get("text") or "").strip()
    summary = str(record.get("summary") or record.get("description") or "").strip()
    published = str(record.get("published") or "").strip()
    source = str(record.get("source") or record.get("url") or "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if published:
        parts.append(f"Published: {published}")
    if source:
        parts.append(f"Source: {source}")
    return "\n".join(parts).strip()


def build_rag_documents(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for index, record in enumerate(records):
        text = record_to_text(record)
        if not text:
            continue
        title = str(record.get("title") or f"record-{index}").strip()
        docs.append({
            "id": f"realtime-{index}",
            "text": text,
            "metadata": {
                "title": title,
                "url": str(record.get("url") or ""),
                "source": str(record.get("source") or ""),
                "published": str(record.get("published") or ""),
                "dataset": "realtime_public_rss",
            },
        })
    return docs


def build_rag_instruction_examples(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    examples = []
    for record in records:
        title = str(record.get("title") or "").strip()
        summary = str(record.get("summary") or "").strip()
        if not title:
            continue

        knowledge = record_to_text(record)
        answer = summary or title
        examples.append({
            "instruction": "Answer using only the retrieved knowledge. If the answer is not present, say you do not have enough information.",
            "input": f"Question: What is the main point of this current item?\n\nRetrieved Knowledge:\n{knowledge}",
            "output": answer,
        })
        examples.append({
            "instruction": "Create a concise enterprise briefing from the retrieved knowledge.",
            "input": f"Retrieved Knowledge:\n{knowledge}",
            "output": f"Briefing: {answer}",
        })
        examples.append({
            "instruction": "Extract source details from the retrieved knowledge.",
            "input": f"Retrieved Knowledge:\n{knowledge}",
            "output": (
                f"Title: {title}\n"
                f"Published: {record.get('published') or 'Not provided'}\n"
                f"Source: {record.get('source') or record.get('url') or 'Not provided'}"
            ),
        })
    return examples


def merge_unique_examples(base_examples: List[Dict[str, str]], new_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged = []
    seen = set()
    for example in base_examples + new_examples:
        key = (
            example.get("instruction", "").strip(),
            example.get("input", "").strip(),
            example.get("output", "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "output": example.get("output", ""),
        })
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RAG documents and RAG-style LoRA training examples from collected real-time data."
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=Path("training/examples/realtime_raw.json"),
        help="Raw collector JSON from collect_realtime_data.py.",
    )
    parser.add_argument(
        "--rag-docs-output",
        type=Path,
        default=Path("training/examples/realtime_rag_documents.json"),
        help="Output JSON document list for Chroma indexing.",
    )
    parser.add_argument(
        "--rag-training-output",
        type=Path,
        default=Path("training/examples/rag_fine_tune_dataset.json"),
        help="Output instruction JSON with RAG-grounded examples.",
    )
    parser.add_argument(
        "--base-training-data",
        type=Path,
        default=Path("training/examples/fine_tune_demo.json"),
        help="Existing instruction dataset to merge with RAG examples.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path("training/examples/fine_tune_realtime_rag.json"),
        help="Merged dataset for LoRA fine-tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.raw_input)
    rag_docs = build_rag_documents(records)
    rag_examples = build_rag_instruction_examples(records)
    base_examples = load_json(args.base_training_data)
    merged_examples = merge_unique_examples(base_examples, rag_examples)

    write_json(args.rag_docs_output, rag_docs)
    write_json(args.rag_training_output, rag_examples)
    write_json(args.merged_output, merged_examples)

    print(f"Loaded records: {len(records)}")
    print(f"RAG documents: {len(rag_docs)} -> {args.rag_docs_output}")
    print(f"RAG training examples: {len(rag_examples)} -> {args.rag_training_output}")
    print(f"Merged fine-tune examples: {len(merged_examples)} -> {args.merged_output}")


if __name__ == "__main__":
    main()
