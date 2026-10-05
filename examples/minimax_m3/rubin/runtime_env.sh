#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Set MSA_ROOT to the patched checkout and CUDA_HOME to CUDA 13.5.7,
# then source this file in every server/worker process environment.
# Build this TRT-LLM branch with --cuda_architectures 107-real and
# --nvrtc_dynamic_linking; fetch the branch's Git LFS assets first.
# Preserve CUDA_HOME explicitly when entering a Pyxis container.
# Launch one TP2 context worker and two TP4 generation workers with
# the matching ctx-<dtype>.yaml and gen-<dtype>.yaml configs.
# For the saved benchmark only, separately export
# TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=1.78. Leave it unset for accuracy.

export MSA_ROOT="${MSA_ROOT:?Set MSA_ROOT to the patched MSA checkout}"
export CUDA_HOME="${CUDA_HOME:?Set CUDA_HOME to the CUDA 13.5.7 toolkit}"
export PYTHONPATH="$MSA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUTLASS_ROOT="$MSA_ROOT/third_party/cutlass"
export OMP_NUM_THREADS=1
export TRTLLM_MSA_NVDEV_PREFILL=1
export FMHA_SM100_ALLOW_JIT=1
export TRTLLM_ENABLE_PDL=1
export TRTLLM_KV_TRANSFER_NUM_THREADS=4
export TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP=0
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1
export NCCL_GRAPH_MIXING_SUPPORT=0
export MIMALLOC_PURGE_DELAY=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRTLLM_SERVE_ENABLE_MSGSPEC=1
export TRTLLM_SERVE_CHAT_TEMPLATE=auto
export TRTLLM_SERVE_TOOL_PARSER=minimax_m3
export MEGAMOE_COMBINE_FORMAT=bf16
export MEGAMOE_TACTIC_AUTOTUNE=0
export TRTLLM_NIXL_BOUNCE_MAX_CHUNK_SIZE_BYTES=32MiB
export TRTLLM_NIXL_BOUNCE_MAX_INFLIGHT_CHUNKS_PER_REQUEST=8
export TRTLLM_NIXL_BOUNCE_COPY_STREAM_COUNT=8
export TRTLLM_NIXL_BOUNCE_SCATTER_WORKER_COUNT=4
export TRTLLM_NIXL_BOUNCE_MIN_DESCRIPTOR_COUNT=1024
export TRTLLM_NIXL_BOUNCE_MAX_AVERAGE_DESCRIPTOR_SIZE_BYTES=16KiB
export TRTLLM_NIXL_BOUNCE_ENABLE_EAGER_GATHER=1
export TRTLLM_NIXL_BOUNCE_USE_NIXL_NOTIFICATIONS=0
export TRTLLM_NIXL_BOUNCE_USE_ZERO_COPY_ARGUMENTS=1
# Native bounce activation and arena sizing are in both worker YAMLs.
# The two old enable/arena environment variables do not activate this build.
# Context workers additionally used:
# TLLM_ADP_ROUTER_MATCH_RATE_THRESHOLD=0.10 TRTLLM_TORCH_COMPILE_CONTEXT_ONLY=1
