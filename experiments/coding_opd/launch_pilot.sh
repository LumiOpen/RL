#!/bin/bash
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
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export TML_OPD_RUN_ID=${TML_OPD_RUN_ID:-coding-seed42}
export TML_OPD_CONFIG=${TML_OPD_CONFIG:-experiments/coding_opd/opd.yaml}
exec sbatch --parsable --nodes=2 --time=12:00:00 --job-name=coding-opd-pilot \
 experiments/tml_opd_replication/opd_multinode.sbatch --audit \
 "logger.wandb.name=$TML_OPD_RUN_ID" "logger.wandb.id=$TML_OPD_RUN_ID" "$@"
