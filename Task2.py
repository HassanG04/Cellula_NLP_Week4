import argparse
import json
from pathlib import Path

from assistants import FileRAGSystem


def main():
    parser = argparse.ArgumentParser(description="Ingest text and ask source-grounded questions")
    parser.add_argument("--store", type=Path, default=Path("data/chunks.json"))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--question")
    args = parser.parse_args()
    rag = FileRAGSystem(args.store)
    if args.file:
        print(json.dumps({"chunks_added": rag.add_documents_from_file(args.file)}))
    if args.question:
        print(json.dumps(rag.ask(args.question), indent=2))


if __name__ == "__main__":
    main()
