"""Interactive CLI for reviewing and re-labeling seed dataset entries."""

import argparse
import json
import logging

logger = logging.getLogger(__name__)


def review_dataset(input_path: str, output_path: str):
    """Interactively review and label entries from a seed dataset JSONL."""
    entries = []
    with open(input_path) as f:
        for line in f:
            entries.append(json.loads(line))

    reviewed = []
    for i, entry in enumerate(entries):
        config = entry["config"]
        current_label = entry.get("label", "UNLABELED")

        print(f"\n{'=' * 60}")
        print(f"Entry {i + 1}/{len(entries)} | Current label: {current_label}")
        print(f"Name: {config.get('name', '?')}")
        print(f"Archetype: {config.get('tension_archetype', '?')}")
        print(f"Context: {config.get('morebench_context', '?')}")
        print(f"Premise: {config.get('premise', '?')[:200]}...")
        for agent in config.get("agents", []):
            print(f"  Agent: {agent.get('name')} ({agent.get('philosophy', 'none')})")
            print(f"    Goal: {agent.get('goal', '?')[:100]}")
        print(f"Schema valid: {entry.get('schema_valid', '?')}")
        print(f"{'=' * 60}")

        while True:
            choice = input(f"Label [G]ood / [B]ad / [S]kip / [Q]uit (current={current_label}): ").strip().upper()
            if choice in ("G", "GOOD"):
                entry["label"] = "GOOD"
                break
            elif choice in ("B", "BAD"):
                entry["label"] = "BAD"
                break
            elif choice in ("S", "SKIP", ""):
                break  # Keep current label
            elif choice in ("Q", "QUIT"):
                # Save what we have
                _save(output_path, entries[:i] + entries[i:])
                print(f"Saved {len(entries)} entries to {output_path}")
                return

        reviewed.append(entry)

    _save(output_path, entries)
    print(f"\nReview complete. Saved {len(entries)} entries to {output_path}")


def _save(path: str, entries: list[dict]):
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(prog="aurelius-seed-labeler", description="Review and label seed dataset")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file (default: overwrite input)")
    args = parser.parse_args()

    output = args.output or args.input
    review_dataset(args.input, output)


if __name__ == "__main__":
    main()
