# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import NHDHeadMismatchMapper
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    KVCachePageTable,
    MapperKind,
    PhysicalPool,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar


class _CudaAllocation:
    """Owning cudaMalloc allocation exposed as a non-owning torch byte view.

    Direct CUDA allocation deliberately bypasses PyTorch expandable segments.
    Large expandable-segment suballocations may start inside a VMM chunk, while
    the C++ NIXL registration path requires a descriptor to start at an
    allocation boundary.
    """

    def __init__(self, size: int, device_id: int) -> None:
        self.device_id = device_id
        self.ptr = 0
        self.tensor: torch.Tensor | None = None
        with torch.cuda.device(device_id):
            error, ptr = cudart.cudaMalloc(size)
            if error != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(
                    f"cudaMalloc failed for {size} NHD staging bytes on device {device_id}: {error}"
                )
            self.ptr = int(ptr)
            self.tensor = convert_to_torch_tensor(TensorWrapper(self.ptr, DataType.INT8, [size]))

    @property
    def view(self) -> torch.Tensor:
        if self.tensor is None:
            raise RuntimeError("NHD staging allocation has already been freed")
        return self.tensor

    def close(self) -> None:
        if self.ptr == 0:
            return
        self.tensor = None
        with torch.cuda.device(self.device_id):
            error = cudart.cudaFree(self.ptr)[0]
        self.ptr = 0
        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaFree failed for NHD staging allocation: {error}")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must not mask an active exception during worker init.
            pass


class NHDStagingBufferLease:
    """Exclusive byte-range lease within a preregistered staging arena."""

    def __init__(
        self,
        manager: NHDStagingBufferManager,
        kind: str,
        index: int,
        offset: int,
        size: int,
        tensor: torch.Tensor,
    ) -> None:
        self._manager = manager
        self._kind = kind
        self._index = index
        self._offset = offset
        self._size = size
        self.tensor = tensor
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release(self._kind, self._index, self._offset, self._size)


class NHDStagingBufferManager:
    """C++-style preregistered send/receive buffers for NHD formatting.

    NIXL agent metadata includes the memory registrations that exist when the
    peer descriptor is serialized. Consequently these buffers are allocated
    before rank-info exchange and reused with exclusive leases per request.
    """

    def __init__(
        self,
        page_table: KVCachePageTable,
        device_id: int,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "TRTLLM_NHD_DISAGG_STAGING requires a positive "
                "cache_transceiver_config.max_tokens_in_buffer"
            )
        self.device_id = device_id
        self.capacity_bytes = self._capacity_bytes(page_table, max_tokens)
        if self.capacity_bytes == 0:
            self._allocations: dict[str, list[_CudaAllocation]] = {"send": [], "recv": []}
            self._buffers: dict[str, list[torch.Tensor]] = {"send": [], "recv": []}
            self._free_ranges: dict[str, list[list[tuple[int, int]]]] = {
                "send": [],
                "recv": [],
            }
            self._condition = threading.Condition()
            return
        send_count = int(os.environ.get("TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM", "1"))
        recv_count = int(os.environ.get("TRTLLM_KVCACHE_RECV_BUFFER_COUNT", "2"))
        if send_count <= 0 or recv_count <= 0:
            raise ValueError(
                "NHD staging send/receive buffer counts must be positive: "
                f"send={send_count}, receive={recv_count}"
            )

        self._allocations = {
            "send": [_CudaAllocation(self.capacity_bytes, device_id) for _ in range(send_count)],
            "recv": [_CudaAllocation(self.capacity_bytes, device_id) for _ in range(recv_count)],
        }
        self._buffers = {
            kind: [allocation.view for allocation in allocations]
            for kind, allocations in self._allocations.items()
        }
        self._free_ranges = {
            kind: [[(0, self.capacity_bytes)] for _ in buffers]
            for kind, buffers in self._buffers.items()
        }
        self._condition = threading.Condition()
        logger.info(
            "Allocated preregistered NHD staging buffers: "
            f"capacity={self.capacity_bytes} bytes, send={send_count}, receive={recv_count}"
        )

    @staticmethod
    def _capacity_bytes(page_table: KVCachePageTable, max_tokens: int) -> int:
        tokens_per_block = page_table.tokens_per_block
        if tokens_per_block <= 0:
            raise ValueError(f"Invalid tokens_per_block for NHD staging: {tokens_per_block}")
        bytes_per_block = 0
        for layer_group in page_table.layer_groups:
            if not isinstance(layer_group, AttentionLayerGroup):
                continue
            for view in layer_group.pool_views:
                if view.mapper_kind != MapperKind.NHD:
                    continue
                if view.bytes_per_region is None:
                    raise ValueError(
                        "NHD staging requires explicit bytes_per_region for every NHD pool"
                    )
                bytes_per_block += view.bytes_per_region
        if bytes_per_block == 0:
            return 0
        # Match CacheTransBufferManager: round up to blocks and reserve one
        # additional block for speculative/draft-token destination capacity.
        max_blocks = (max_tokens + tokens_per_block - 1) // tokens_per_block + 1
        return max_blocks * bytes_per_block

    @property
    def memory_descs(self) -> list[tuple[int, int, int, str]]:
        return [
            (
                tensor.data_ptr(),
                tensor.numel(),
                self.device_id,
                f"nhd_staging_{kind}_{index}",
            )
            for kind, buffers in self._buffers.items()
            for index, tensor in enumerate(buffers)
        ]

    def close(self) -> None:
        with self._condition:
            self._buffers = {"send": [], "recv": []}
            allocations = self._allocations
            self._allocations = {"send": [], "recv": []}
        for buffers in allocations.values():
            for allocation in buffers:
                allocation.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def acquire(self, kind: str, size: int) -> NHDStagingBufferLease:
        if kind not in self._buffers:
            raise ValueError(f"Unknown NHD staging buffer kind: {kind}")
        if size <= 0 or size > self.capacity_bytes:
            raise ValueError(
                "NHD staging request exceeds preregistered capacity: "
                f"requested={size}, capacity={self.capacity_bytes}, kind={kind}"
            )
        with self._condition:
            self._condition.wait_for(lambda: self._find_free_range(kind, size) is not None)
            found = self._find_free_range(kind, size)
            if found is None:
                raise RuntimeError("NHD staging allocator woke without a suitable free range")
            index, range_index, offset = found
            range_offset, range_size = self._free_ranges[kind][index].pop(range_index)
            remaining = range_size - size
            if remaining:
                self._free_ranges[kind][index].insert(range_index, (range_offset + size, remaining))
        return NHDStagingBufferLease(
            self,
            kind,
            index,
            offset,
            size,
            self._buffers[kind][index][offset : offset + size],
        )

    def _find_free_range(self, kind: str, size: int) -> tuple[int, int, int] | None:
        for buffer_index, ranges in enumerate(self._free_ranges[kind]):
            for range_index, (offset, available) in enumerate(ranges):
                if available >= size:
                    return buffer_index, range_index, offset
        return None

    def _release(self, kind: str, index: int, offset: int, size: int) -> None:
        with self._condition:
            ranges = self._free_ranges[kind][index]
            ranges.append((offset, size))
            ranges.sort()
            merged: list[tuple[int, int]] = []
            for current_offset, current_size in ranges:
                if not merged:
                    merged.append((current_offset, current_size))
                    continue
                previous_offset, previous_size = merged[-1]
                previous_end = previous_offset + previous_size
                if current_offset < previous_end:
                    raise RuntimeError(
                        f"Overlapping release for NHD staging buffer {kind}[{index}]"
                    )
                if current_offset == previous_end:
                    merged[-1] = (previous_offset, previous_size + current_size)
                else:
                    merged.append((current_offset, current_size))
            self._free_ranges[kind][index] = merged
            self._condition.notify_all()


def _pool_bytes(pool: PhysicalPool) -> torch.Tensor:
    size = pool.num_slots * pool.slot_bytes
    return convert_to_torch_tensor(TensorWrapper(pool.base_address, DataType.INT8, [size])).view(
        pool.num_slots, pool.slot_bytes
    )


def _validate_logical_view(pool: PhysicalPool, view: PoolView, *, label: str) -> int:
    region_bytes = view.bytes_per_region
    if region_bytes is None:
        raise ValueError(f"NHD staging requires an explicit {label} bytes_per_region")
    if view.byte_offset < 0 or region_bytes <= 0:
        raise ValueError(
            f"Invalid {label} NHD logical range: offset={view.byte_offset}, bytes={region_bytes}"
        )
    if view.byte_offset + region_bytes > pool.slot_bytes:
        raise ValueError(
            f"{label.capitalize()} NHD logical range exceeds its physical slot: "
            f"offset={view.byte_offset}, bytes={region_bytes}, "
            f"slot_bytes={pool.slot_bytes}"
        )
    return region_bytes


@dataclass(frozen=True)
class NHDStagingChunk:
    """One logical NHD pool transfer packed into a contiguous byte range."""

    src_pool: PhysicalPool
    src_view: PoolView
    dst_pool: PhysicalPool
    dst_view: PoolView
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    mapper: NHDHeadMismatchMapper
    staging_offset: int

    @property
    def payload_bytes(self) -> int:
        mapper = self.mapper
        return (
            self.src_block_ids.size
            * mapper.transfer_layers
            * mapper.src_buffers_per_layer
            * mapper.tokens_per_block
            * mapper.contiguous_heads
            * mapper.src_bytes_per_token_head
        )

    def _src_logical_view(self) -> torch.Tensor:
        mapper = self.mapper
        region_bytes = _validate_logical_view(self.src_pool, self.src_view, label="source")
        begin = self.src_view.byte_offset
        end = begin + region_bytes
        return _pool_bytes(self.src_pool)[:, begin:end].view(
            self.src_pool.num_slots,
            mapper.src_pool_num_layers,
            mapper.src_buffers_per_layer,
            mapper.tokens_per_block,
            mapper.src_heads,
            mapper.src_bytes_per_token_head,
        )

    def _dst_logical_view(self) -> torch.Tensor:
        mapper = self.mapper
        region_bytes = _validate_logical_view(self.dst_pool, self.dst_view, label="destination")
        begin = self.dst_view.byte_offset
        end = begin + region_bytes
        return _pool_bytes(self.dst_pool)[:, begin:end].view(
            self.dst_pool.num_slots,
            mapper.dst_pool_num_layers,
            mapper.dst_buffers_per_layer,
            mapper.tokens_per_block,
            mapper.dst_heads,
            mapper.dst_bytes_per_token_head,
        )

    def pack_into(self, staging: torch.Tensor) -> None:
        mapper = self.mapper
        block_ids = torch.as_tensor(
            self.src_block_ids,
            dtype=torch.long,
            device=staging.device,
        )
        # Slice heads before gathering blocks so DEP->TEP does not materialize
        # the unused source heads in an intermediate tensor. This mirrors the
        # C++ split kernel, which writes only the selected head range.
        selected_heads = self._src_logical_view()[
            :,
            mapper.src_layer_offset : mapper.src_layer_offset + mapper.transfer_layers,
            :,
            :,
            mapper.src_head_index : mapper.src_head_index + mapper.contiguous_heads,
            :,
        ]
        selected = selected_heads.index_select(0, block_ids)
        begin = self.staging_offset
        staging[begin : begin + self.payload_bytes].copy_(selected.reshape(-1))

    def scatter_from(self, staging: torch.Tensor) -> None:
        mapper = self.mapper
        block_ids = torch.as_tensor(
            self.dst_block_ids,
            dtype=torch.long,
            device=staging.device,
        )
        begin = self.staging_offset
        packed = staging[begin : begin + self.payload_bytes].view(
            self.dst_block_ids.size,
            mapper.transfer_layers,
            mapper.dst_buffers_per_layer,
            mapper.tokens_per_block,
            mapper.contiguous_heads,
            mapper.dst_bytes_per_token_head,
        )
        dst = self._dst_logical_view()
        dst[
            block_ids,
            mapper.dst_layer_offset : mapper.dst_layer_offset + mapper.transfer_layers,
            :,
            :,
            mapper.dst_head_index : mapper.dst_head_index + mapper.contiguous_heads,
            :,
        ] = packed


class NHDStagingPlan:
    """Pack/scatter a collection of NHD head-mismatch pool mappings."""

    def __init__(self, chunks: list[NHDStagingChunk]):
        self.chunks = chunks
        self.payload_bytes = sum(chunk.payload_bytes for chunk in chunks)
        expected_offset = 0
        for chunk in chunks:
            if chunk.staging_offset != expected_offset:
                raise ValueError(
                    "NHD staging chunks must be tightly packed in order: "
                    f"expected offset {expected_offset}, got {chunk.staging_offset}"
                )
            expected_offset += chunk.payload_bytes

    def allocate(self, device: torch.device | int | str) -> torch.Tensor:
        if isinstance(device, int):
            device = torch.device("cuda", device)
        return torch.empty(self.payload_bytes, dtype=torch.int8, device=device)

    def _validate_buffer(self, staging: torch.Tensor) -> None:
        if staging.dtype != torch.int8 or not staging.is_cuda or not staging.is_contiguous():
            raise ValueError(
                "NHD staging buffer must be a contiguous CUDA int8 tensor; "
                f"got dtype={staging.dtype}, device={staging.device}, "
                f"contiguous={staging.is_contiguous()}"
            )
        if staging.numel() < self.payload_bytes:
            raise ValueError(
                f"NHD staging buffer is too small: {staging.numel()} < {self.payload_bytes}"
            )

    def pack_into(self, staging: torch.Tensor) -> None:
        self._validate_buffer(staging)
        for chunk in self.chunks:
            chunk.pack_into(staging)

    def scatter_from(self, staging: torch.Tensor) -> None:
        self._validate_buffer(staging)
        for chunk in self.chunks:
            chunk.scatter_from(staging)


def build_nhd_staging_plan(
    registrar: PeerRegistrar,
    receiver_ri: RankInfo,
    src_block_ids_per_groups: list[np.ndarray] | None,
    dst_block_ids_per_groups: list[np.ndarray],
) -> NHDStagingPlan:
    """Build the canonical sender-to-receiver NHD staging order.

    ``src_block_ids_per_groups`` may be omitted by the receiver because only
    block count and destination IDs are needed for scatter. The sender always
    supplies its aligned source block IDs.
    """
    src_page_table = registrar.self_extractor.page_table
    dst_page_table = receiver_ri.page_table
    if dst_page_table is None:
        raise ValueError("NHD staging requires a receiver page table")
    sender_ri = registrar.self_rank_info
    if sender_ri.cp_size != 1 or receiver_ri.cp_size != 1:
        raise NotImplementedError(
            "NHD staging currently supports CP=1 only: "
            f"sender_cp={sender_ri.cp_size}, receiver_cp={receiver_ri.cp_size}"
        )

    peer_overlap = registrar.get_peer_overlap(receiver_ri, receiver_ri.dp_rank)
    chunks: list[NHDStagingChunk] = []
    staging_offset = 0
    pool_mapping = registrar.get_pool_mapping(receiver_ri)
    for (src_lg, src_pi), (dst_lg, dst_pi) in sorted(pool_mapping.items()):
        if not registrar.should_send_pool(
            peer_overlap,
            receiver_ri,
            src_lg,
            src_pi,
        ):
            continue
        mapper = registrar.get_kv_map(
            receiver_ri,
            (src_lg, src_pi),
            (dst_lg, dst_pi),
        )
        if not isinstance(mapper, NHDHeadMismatchMapper):
            continue

        dst_block_ids = dst_block_ids_per_groups[dst_lg]
        if src_block_ids_per_groups is None:
            src_block_ids = np.arange(dst_block_ids.size, dtype=np.int64)
        else:
            src_block_ids = src_block_ids_per_groups[src_lg]
        if src_block_ids.size != dst_block_ids.size:
            raise ValueError(
                "NHD staging requires aligned source/destination block counts: "
                f"source={src_block_ids.size}, destination={dst_block_ids.size}, "
                f"source_layer_group={src_lg}, destination_layer_group={dst_lg}"
            )

        src_layer_group = src_page_table.layer_groups[src_lg]
        dst_layer_group = dst_page_table.layer_groups[dst_lg]
        src_view = src_layer_group.pool_views[src_pi]
        dst_view = dst_layer_group.pool_views[dst_pi]
        chunk = NHDStagingChunk(
            src_pool=get_physical_pool(src_page_table, src_lg, src_view.pool_idx),
            src_view=src_view,
            dst_pool=get_physical_pool(dst_page_table, dst_lg, dst_view.pool_idx),
            dst_view=dst_view,
            src_block_ids=np.asarray(src_block_ids, dtype=np.int64).copy(),
            dst_block_ids=np.asarray(dst_block_ids, dtype=np.int64).copy(),
            mapper=mapper,
            staging_offset=staging_offset,
        )
        chunks.append(chunk)
        staging_offset += chunk.payload_bytes

    return NHDStagingPlan(chunks)
