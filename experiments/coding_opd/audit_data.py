# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit held-out prompts against actual SFT inputs and prepare disjoint TACO prompts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re

from prompts import format_prompt

ROOT = Path(__file__).resolve().parent
TOKENIZER = None
INDEX = None
GRAMS = None


def normalize(text):
    return re.findall(r"\w+", text.lower())


def grams(text):
    words = normalize(text)
    return {tuple(words[i : i + 13]) for i in range(len(words) - 12)}


def initialize():
    global TOKENIZER, INDEX, GRAMS
    from transformers import AutoTokenizer

    protocol = json.loads((ROOT / "protocol.json").read_text())
    TOKENIZER = AutoTokenizer.from_pretrained(protocol["base_model"])
    TOKENIZER.chat_template = (
        ROOT.parent / "tml_opd_replication/qwen3_historical.jinja"
    ).read_text()
    GRAMS = {
        row["id"]: grams(row["question"])
        for row in map(
            json.loads, (ROOT / "data/eval_candidates.jsonl").read_text().splitlines()
        )
    }
    INDEX = defaultdict(set)
    for key, parts in GRAMS.items():
        if len(parts) < 4:
            raise ValueError(f"Evaluation prompt too short to audit: {key}")
        for part in parts:
            INDEX[part].add(key)


def overlaps(text):
    counts = Counter(key for part in grams(text) for key in INDEX.get(part, ()))
    return {
        key: count / len(GRAMS[key])
        for key, count in counts.items()
        if count >= 4 and count / len(GRAMS[key]) >= 0.5
    }


def audit_shard(path):
    import numpy as np
    import pyarrow as pa

    hits, rows = [], 0
    with pa.memory_map(str(path), "r") as file:
        reader = pa.ipc.open_stream(file)
        for batch in reader:
            ids = batch.column("input_ids")
            masks = batch.column("loss_mask")
            id_offsets, mask_offsets = ids.offsets.to_numpy(), masks.offsets.to_numpy()
            id_values, mask_values = ids.values.to_numpy(), masks.values.to_numpy()
            prompts = []
            for i in range(len(batch)):
                mask = mask_values[mask_offsets[i] : mask_offsets[i + 1]]
                trained = np.flatnonzero(mask)
                stop = int(trained[0]) if len(trained) else len(mask)
                prompts.append(id_values[id_offsets[i] : id_offsets[i] + stop].tolist())
            for i, text in enumerate(
                TOKENIZER.batch_decode(prompts, skip_special_tokens=True)
            ):
                found = overlaps(text)
                if found:
                    hits.append({"shard": path.name, "row": rows + i, "matches": found})
            rows += len(batch)
    return {"shard": path.name, "rows": rows, "hits": hits}


def prepare_taco():
    import pyarrow.parquet as pq

    seen, stats = set(), Counter()
    target = ROOT / "data/taco_historical.jsonl"
    with target.with_suffix(".tmp").open("w") as out:
        for path in sorted((ROOT / "data/taco").glob("*.parquet")):
            table = pq.read_table(path, columns=["question", "starter_code"])
            for row_index, row in enumerate(table.to_pylist()):
                stats["source_rows"] += 1
                question = row["question"]
                digest = hashlib.sha256(
                    " ".join(normalize(question)).encode()
                ).hexdigest()
                if digest in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(digest)
                if overlaps(question):
                    stats["evaluation_overlaps"] += 1
                    continue
                starter = row["starter_code"] or ""
                prompt = format_prompt(question, starter)
                if len(TOKENIZER.encode(prompt)) > 1024:
                    stats["over_length"] += 1
                    continue
                tokens = TOKENIZER.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
                assert len(tokens) + 4096 <= 5184
                out.write(
                    json.dumps(
                        {
                            "idx": stats["kept"],
                            "question": prompt,
                            "input_ids": tokens,
                            "source_shard": path.name,
                            "source_row": row_index,
                            "question_sha256": digest,
                        }
                    )
                    + "\n"
                )
                stats["kept"] += 1
                stats["max_rendered_tokens"] = max(
                    stats["max_rendered_tokens"], len(tokens)
                )
            print("TACO", path.name, dict(stats), flush=True)
    target.with_suffix(".tmp").replace(target)
    (ROOT / "data/taco_manifest.json").write_text(
        json.dumps(dict(stats), indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    paths = sorted(Path(protocol["sft_data"]).glob("tokens-*.arrow"))
    assert len(paths) == 375, len(paths)
    audit = ROOT / "data/sft_overlap_shards.jsonl"
    completed = (
        {row["shard"]: row for row in map(json.loads, audit.read_text().splitlines())}
        if audit.exists()
        else {}
    )
    with (
        ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool,
        audit.open("a", buffering=1) as out,
    ):
        for result in pool.map(
            audit_shard, [p for p in paths if p.name not in completed]
        ):
            completed[result["shard"]] = result
            out.write(json.dumps(result) + "\n")
            print("SFT_AUDIT", len(completed), "/", len(paths), flush=True)
    excluded = sorted(
        {
            key
            for row in completed.values()
            for hit in row["hits"]
            for key in hit["matches"]
        }
    )
    report = {
        "sft_rows_checked": sum(row["rows"] for row in completed.values()),
        "excluded_problem_ids": excluded,
        "criterion": "At least 4 shared unique normalized 13-word ngrams covering at least 50% of evaluation-question ngrams; SFT unmasked input only.",
    }
    assert report["sft_rows_checked"] == 384000
    (ROOT / "data/overlap_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    candidates = [
        json.loads(line)
        for line in (ROOT / "data/eval_candidates.jsonl").read_text().splitlines()
    ]
    (ROOT / "data/eval_clean.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n" for row in candidates if row["id"] not in excluded
        )
    )
    initialize()
    prepare_taco()
    print("AUDIT_COMPLETE", report, flush=True)


if __name__ == "__main__":
    main()
