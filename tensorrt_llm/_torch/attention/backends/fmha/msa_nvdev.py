# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cache-layout adapter for MiniMax MSA's public inference wrappers."""

from __future__ import annotations

import math

import torch


def virtual_contiguous_pages(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Expose an interleaved HND cache as contiguous virtual physical pages.

    Args:
        cache: A four-dimensional CUDA cache with contiguous inner pages.
            Dtypes and page dimensions are validated by the consuming wrapper.

    Returns:
        A zero-copy contiguous view and the multiplier for physical page IDs.
        Gaps between real pages become unreferenced virtual pages. Logical
        TopK indices and token lengths remain unchanged.
    """
    if cache.ndim != 4 or cache.shape[0] <= 0:
        raise ValueError("MSA cache must contain [pages, heads, tokens, dimension]")
    payload = math.prod(cache.shape[1:])
    if payload <= 0 or not cache[0].is_contiguous():
        raise ValueError("MSA virtual pages require contiguous inner HND pages")
    page_stride = cache.stride(0)
    if page_stride < payload or page_stride % payload:
        raise ValueError("MSA page stride must be a positive whole number of page payloads")
    multiplier = page_stride // payload
    virtual_count = (cache.shape[0] - 1) * multiplier + 1
    # Retain this tensor's storage offset: K and V start at different planes.
    # The final virtual page ends at the original final page, so the view
    # never extends beyond the original cache's storage bounds.
    view = cache.as_strided(
        (virtual_count, *cache.shape[1:]),
        (payload, *cache.stride()[1:]),
    )
    return view, multiplier


def map_cache_pages(
    page_table: torch.Tensor, *caches: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Adapt one authoritative physical-page table and matching cache planes."""
    if not caches:
        raise ValueError("At least one cache plane is required")
    views_and_factors = tuple(virtual_contiguous_pages(cache) for cache in caches)
    factor = views_and_factors[0][1]
    if any(value != factor for _, value in views_and_factors):
        raise ValueError("K/V and scale planes must share the virtual-page multiplier")
    if any(cache.shape[0] != caches[0].shape[0] for cache in caches):
        raise ValueError("K/V and scale planes must share the original page capacity")
    mapped = page_table
    if factor != 1:
        mapped = torch.where(page_table >= 0, page_table * factor, page_table)
    return mapped, tuple(view for view, _ in views_and_factors)


def run_fp8_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    total_k: int,
    total_rows: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    out: torch.Tensor,
) -> None:
    """Run native FP8 MSA prefill, owning a fresh plan for this layer's TopK."""
    from inference.msa_v1.attention.prefill.q8kv8 import BatchPrefillWithPagedKVCacheWrapper

    mapped_pages, (k_pages, v_pages) = map_cache_pages(page_table, k_cache, v_cache)
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        mapped_pages,
        num_q_heads=q.shape[1],
        num_kv_heads=k_cache.shape[1],
        total_k=total_k,
        total_rows=total_rows,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
    )
    wrapper.run(q, (k_pages, v_pages), out=out)


def run_nvfp4_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    total_k: int,
    total_rows: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    out: torch.Tensor,
) -> None:
    """Adapt TRT's calibrated NVFP4 storage to the extended public wrapper."""
    from inference.msa_v1.attention.prefill.q8kv4 import BatchPrefillWithPagedKVCacheWrapper

    # Data and scale pools can have different page pitches: the index-K
    # plane joins the packed-data pool only when their byte sizes match.
    # Keep the authoritative slot table and original zero-copy views; the
    # opt-in native extension reads each plane's own dim-0 stride.
    wrapper = BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        total_k=total_k,
        total_rows=total_rows,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
    )
    wrapper.run(
        q.to(torch.float8_e4m3fn),
        (k_cache.view(torch.uint8), v_cache.view(torch.uint8)),
        kv_cache_sf=(k_scale.view(torch.float8_e4m3fn), v_scale.view(torch.float8_e4m3fn)),
        allow_strided_cache=True,
        kv_cache_global_sf=(k_global_scale, v_global_scale),
        v_scale_layout="swizzle4x4",
        out=out,
    )
