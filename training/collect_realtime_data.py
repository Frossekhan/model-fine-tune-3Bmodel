import argparse
import json
import re
import ssl
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_FEEDS = [
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://hnrss.org/frontpage",
]

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = HTML_TAG_PATTERN.sub(" ", value or "")
    value = value.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    return SPACE_PATTERN.sub(" ", value).strip()


def fetch_feed(url: str, timeout: int, allow_insecure_ssl: bool = False) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "enterprise-ai-training-data-collector/1.0"},
    )
    context = ssl._create_unverified_context() if allow_insecure_ssl else None
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def parse_rss_items(feed_xml: bytes, source_url: str) -> Iterable[Dict[str, str]]:
    root = ET.fromstring(feed_xml)
    channel_items = root.findall(".//item")
    atom_entries = root.findall("{http://www.w3.org/2005/Atom}entry")

    for item in channel_items:
        title = clean_text(item.findtext("title", ""))
        description = clean_text(item.findtext("description", ""))
        link = clean_text(item.findtext("link", ""))
        published = clean_text(item.findtext("pubDate", ""))
        if title:
            yield {
                "title": title,
                "summary": description,
                "url": link,
                "published": published,
                "source": source_url,
            }

    for entry in atom_entries:
        title = clean_text(entry.findtext("{http://www.w3.org/2005/Atom}title", ""))
        summary = clean_text(entry.findtext("{http://www.w3.org/2005/Atom}summary", ""))
        link_node = entry.find("{http://www.w3.org/2005/Atom}link")
        link = clean_text(link_node.attrib.get("href", "")) if link_node is not None else ""
        published = clean_text(entry.findtext("{http://www.w3.org/2005/Atom}updated", ""))
        if title:
            yield {
                "title": title,
                "summary": summary,
                "url": link,
                "published": published,
                "source": source_url,
            }


def to_instruction_record(item: Dict[str, str]) -> Dict[str, str]:
    input_text = item["title"]
    if item.get("summary"):
        input_text = f'{item["title"]}\n\nContext: {item["summary"]}'
    return {
        "instruction": "Summarize this current public item for an enterprise user.",
        "input": input_text,
        "output": item["summary"] or item["title"],
    }


def to_neutral_sentiment_record(item: Dict[str, str]) -> Dict[str, str]:
    return {
        "text": item["title"],
        "label": "neutral",
    }


def collect_records(
    feed_urls: List[str],
    limit: int,
    timeout: int,
    allow_insecure_ssl: bool = False,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    seen_titles = set()
    for feed_url in feed_urls:
        try:
            feed_xml = fetch_feed(feed_url, timeout, allow_insecure_ssl)
            for item in parse_rss_items(feed_xml, feed_url):
                key = item["title"].lower()
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                records.append(item)
                if len(records) >= limit:
                    return records
        except Exception as exc:
            print(f"warning: failed to collect {feed_url}: {exc}", file=sys.stderr)
    return records


def write_json(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_training_json(path: Path, records: List[Dict[str, str]], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "instruction":
        training_records = [to_instruction_record(record) for record in records]
    elif mode == "sentiment-neutral":
        training_records = [to_neutral_sentiment_record(record) for record in records]
    else:
        raise ValueError(f"unsupported mode: {mode}")
    path.write_text(json.dumps(training_records, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect current public RSS items into reviewable training data."
    )
    parser.add_argument(
        "--feed",
        action="append",
        dest="feeds",
        help="RSS/Atom feed URL. Can be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum records to collect.")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Disable SSL certificate verification for local/dev collection only.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("training/examples/realtime_raw.json"),
        help="Path for raw collected records with metadata.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("training/examples/realtime_instruction_dataset.json"),
        help="Path for generated training records.",
    )
    parser.add_argument(
        "--mode",
        choices=["instruction", "sentiment-neutral"],
        default="instruction",
        help="Training dataset format to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feed_urls = args.feeds or DEFAULT_FEEDS
    records = collect_records(
        feed_urls,
        max(args.limit, 1),
        max(args.timeout, 1),
        args.allow_insecure_ssl,
    )
    write_json(args.raw_output, records)
    write_training_json(args.training_output, records, args.mode)
    print(f"Collected {len(records)} records")
    print(f"Raw data: {args.raw_output}")
    print(f"Training data: {args.training_output}")


if __name__ == "__main__":
    main()
