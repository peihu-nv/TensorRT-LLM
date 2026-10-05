#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Usage: bash setup_msa.sh /absolute/path/to/new/msa-checkout
# Native NVFP4 JIT compilation requires CUDA 13.5.7 and CuTe DSL 4.8.
# The saved full-model run used CuTe DSL 4.8.0.dev0; stable 4.8.0 was
# validated separately for component correctness.
msa_destination="${1:?Supply a new MSA checkout directory}"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -e "$msa_destination" ]]; then
    echo "Destination already exists: $msa_destination" >&2
    exit 1
fi
git clone https://github.com/MiniMax-AI/MSA.git "$msa_destination"
git -C "$msa_destination" checkout --detach c8e1013a16a708d2a7656805d5fc26f2b87715d1
gzip -dc "$script_directory/msa-nvfp4-v3.patch.gz" | git -C "$msa_destination" apply --check -
gzip -dc "$script_directory/msa-nvfp4-v3.patch.gz" | git -C "$msa_destination" apply -
git -C "$msa_destination" submodule update --init --recursive third_party/cutlass
test "$(git -C "$msa_destination/third_party/cutlass" rev-parse HEAD)" = dc45f979ae336a235da1676b311f35efeb30149a
echo "MSA is ready at $msa_destination; export MSA_ROOT to this absolute path."
