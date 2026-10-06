# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the same balanced, held-out-audited prompt stream for both pilot arms."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO.parent / "RL"
SOURCES = {
    "math": OLD / "experiments/tml_opd_replication/data/deepmath_historical.jsonl",
    "coding": REPO / "experiments/coding_opd/data/taco_historical.jsonl",
}
HELD_OUT = [
    REPO / "experiments/coding_opd/data/eval_clean.jsonl",
    REPO / "experiments/coding_opd/run_aime_51701/aime_prompts.jsonl",
]


def grams(text):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + 13]) for i in range(len(words) - 12)}


def main():
    index = defaultdict(set)
    sizes, normalized = {}, set()
    for path in HELD_OUT:
        for row in map(json.loads, path.read_text().splitlines()):
            key = row["id"]
            parts = grams(row["question"])
            sizes[key] = len(parts)
            normalized.add(" ".join(re.findall(r"\w+", row["question"].lower())))
            for part in parts:
                index[part].add(key)
    rows_by_domain, manifest = {}, {"seed": 42, "held_out_problems": len(sizes)}
    for domain, source in SOURCES.items():
        rng = random.Random(42)
        reservoir, seen, excluded = [], 0, 0
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                if len(row["input_ids"]) > 1024:
                    continue
                text = row["question"]
                counts = Counter(
                    key for part in grams(text) for key in index.get(part, ())
                )
                overlaps = any(
                    n >= 4 and n >= sizes[key] / 2 for key, n in counts.items()
                )
                if overlaps or " ".join(re.findall(r"\w+", text.lower())) in normalized:
                    excluded += 1
                    continue
                seen += 1
                item = {
                    "agent_ref": {
                        "type": "responses_api_agents",
                        "name": f"{domain}_agent",
                    },
                    "responses_create_params": {
                        "input": [{"role": "user", "content": text}]
                    },
                    "source_idx": row["idx"],
                }
                if len(reservoir) < 320:
                    reservoir.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < 320:
                        reservoir[j] = item
        assert len(reservoir) == 320
        rng.shuffle(reservoir)
        rows_by_domain[domain] = reservoir
        manifest[domain] = {
            "source": str(source),
            "source_sha256": digest.hexdigest(),
            "eligible": seen,
            "overlap_exclusions": excluded,
            "selected": len(reservoir),
            "source_indices": [row["source_idx"] for row in reservoir],
        }
    output = ROOT / "data/mixed_pilot.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for pair in zip(rows_by_domain["math"], rows_by_domain["coding"])
        for row in pair
    ]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["overlap_criterion"] = (
        "Exact normalized match or >=4 shared 13-word ngrams covering >=50% of held-out question ngrams."
    )
    (ROOT / "data/mixed_pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k not in SOURCES}, indent=2))


if __name__ == "__main__":
    main()
