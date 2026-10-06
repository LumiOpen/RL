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

"""Download pinned datasets and render candidate evaluation prompts."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlretrieve

from prompts import format_prompt

ROOT = Path(__file__).resolve().parent


def download(task):
    url, target = task
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        urlretrieve(url, temporary)
        temporary.replace(target)
    print(target.name, target.stat().st_size, flush=True)


def main():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    tasks = []
    for name, repo, filenames in [
        ("lcb", "livecodebench/code_generation_lite", ["test6.jsonl"]),
        (
            "taco",
            "BAAI/TACO",
            [f"ALL/train-{i:05d}-of-00009.parquet" for i in range(9)],
        ),
    ]:
        revision = protocol[name + "_dataset_revision"]
        for filename in filenames:
            tasks.append(
                (
                    f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}",
                    ROOT / "data" / name / Path(filename).name,
                )
            )
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, tasks))
    rows = []
    for line in (ROOT / "data/lcb/test6.jsonl").open():
        row = json.loads(line)
        rows.append(
            {
                "id": row["platform"] + ":" + row["question_id"],
                "question": row["question_content"],
                "prompt": format_prompt(row["question_content"], row["starter_code"]),
                "difficulty": row["difficulty"],
                "contest_date": row["contest_date"],
            }
        )
    assert len(rows) == 175
    assert len({r["id"] for r in rows}) == len(rows)
    target = ROOT / "data/eval_candidates.jsonl"
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if target.exists() and target.read_text() != text:
        raise ValueError("Existing evaluation prompts differ; do not mix protocols")
    target.write_text(text)


if __name__ == "__main__":
    main()
