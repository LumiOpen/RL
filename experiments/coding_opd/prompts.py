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

"""The pinned LiveCodeBench generic user prompt (no system message)."""


def format_prompt(question, starter_code):
    if starter_code:
        instruction = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    else:
        instruction = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
    return (
        f"### Question:\n{question}\n\n### Format: {instruction}\n"
        f"```python\n{starter_code or '# YOUR CODE HERE'}\n```\n\n"
        "### Answer: (use the provided format with backticks)\n\n"
    )
