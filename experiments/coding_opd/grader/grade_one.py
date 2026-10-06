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

"""Run one candidate in an externally isolated process using the pinned LCB checker."""

import contextlib
import json
import sys

import numpy as np
from testing_util import run_test

payload = json.load(sys.stdin)
stream = sys.stdout
with (
    open("/dev/null", "w") as sink,
    contextlib.redirect_stdout(sink),
    contextlib.redirect_stderr(sink),
):
    results, metadata = run_test(
        {"input_output": json.dumps(payload["tests"])}, test=payload["code"], timeout=6
    )
values = [
    int(x)
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
    else bool(x)
    for x in results
]
json.dump(
    {
        "passed": bool(values) and all(x is True for x in values),
        "results": values,
        "error_code": metadata.get("error_code"),
        "error_message": str(metadata.get("error_message", ""))[:1000],
    },
    stream,
)
stream.write("\n")
