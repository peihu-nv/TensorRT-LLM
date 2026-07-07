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

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import NHDHeadMismatchMapper
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.staging import (
    NHDStagingBufferManager,
    NHDStagingChunk,
    NHDStagingPlan,
)
from tensorrt_llm._torch.disaggregation.native.transfer import RecvReqInfo
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    MapperKind,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)


def _rank_info(*, name: str, rank: int, tp_size: int, tp_rank: int, heads: int) -> RankInfo:
    return RankInfo(
        instance_name=name,
        instance_rank=rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=0,
        layer_num_per_pp=[1],
        sender_endpoints=[],
        server_endpoint="",
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=heads,
            tokens_per_block=2,
            dims_per_head=2,
            element_bytes=1,
            enable_attention_dp=False,
            is_mla=False,
        ),
    )


def _pool(tensor: torch.Tensor) -> PhysicalPool:
    return PhysicalPool(
        base_address=tensor.data_ptr(),
        slot_bytes=tensor.shape[1],
        num_slots=tensor.shape[0],
    )


def _view(buffer_bytes: int) -> PoolView:
    return PoolView(
        pool_idx=0,
        buffer_entries=np.array(
            [(0, 0, buffer_bytes), (0, buffer_bytes, buffer_bytes)],
            dtype=BUFFER_ENTRY_DTYPE,
        ),
        pool_role=frozenset({"key", "value"}),
        mapper_kind=MapperKind.NHD,
        byte_offset=0,
        bytes_per_region=2 * buffer_bytes,
    )


def test_recv_req_info_round_trip_preserves_nhd_staging() -> None:
    req = RecvReqInfo(
        sender_req_id=11,
        instance_name="generation",
        instance_rank=3,
        block_ids_per_layer_groups=[np.array([7, 2], dtype=np.int64)],
        unique_rid=19,
        nhd_staging_ptr=0x12340000,
        nhd_staging_size=65536,
    )

    restored = RecvReqInfo.from_bytes(req.to_bytes())

    assert restored.nhd_staging_ptr == req.nhd_staging_ptr
    assert restored.nhd_staging_size == req.nhd_staging_size
    np.testing.assert_array_equal(
        restored.block_ids_per_layer_groups[0], req.block_ids_per_layer_groups[0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nhd_staging_buffer_manager_preregisters_fixed_capacity(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_KVCACHE_SEND_MAX_CONCURRENCY_NUM", "1")
    monkeypatch.setenv("TRTLLM_KVCACHE_RECV_BUFFER_COUNT", "2")
    view = _view(8)
    page_table = KVCachePageTable(
        tokens_per_block=2,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=2,
                local_layers=[LocalLayer(local_layer_id=0, global_layer_id=0)],
                pool_views=[view],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(pools=[PhysicalPool(base_address=0, slot_bytes=16, num_slots=4)])
        ],
    )

    manager = NHDStagingBufferManager(page_table, device_id=0, max_tokens=4)

    # ceil(4 / 2) data blocks plus the C++-compatible extra draft block.
    assert manager.capacity_bytes == 3 * 16
    assert len(manager.memory_descs) == 3
    lease = manager.acquire("send", 32)
    assert lease.tensor.numel() == 32
    assert lease.tensor.data_ptr() in {desc[0] for desc in manager.memory_descs}
    lease.release()

    first = manager.acquire("recv", 20)
    second = manager.acquire("recv", 20)
    assert second.tensor.data_ptr() == first.tensor.data_ptr() + 20
    first.release()
    second.release()
    merged = manager.acquire("recv", manager.capacity_bytes)
    merged.release()

    with pytest.raises(ValueError, match="exceeds preregistered capacity"):
        manager.acquire("recv", manager.capacity_bytes + 1)

    manager.close()
    manager.close()
    assert manager.memory_descs == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "src_tp,src_rank,src_heads,dst_tp,dst_rank,dst_heads,src_head,dst_head",
    [
        pytest.param(1, 0, 2, 2, 1, 1, 1, 0, id="dep_to_tep"),
        pytest.param(2, 1, 1, 1, 0, 2, 0, 1, id="tep_to_dep"),
    ],
)
def test_nhd_staging_pack_and_scatter_head_slice(
    src_tp: int,
    src_rank: int,
    src_heads: int,
    dst_tp: int,
    dst_rank: int,
    dst_heads: int,
    src_head: int,
    dst_head: int,
) -> None:
    bytes_per_buffer_per_head = 4
    src_slot_bytes = 2 * bytes_per_buffer_per_head * src_heads
    dst_slot_bytes = 2 * bytes_per_buffer_per_head * dst_heads
    src = torch.arange(3 * src_slot_bytes, dtype=torch.int8, device="cuda").view(3, src_slot_bytes)
    dst = torch.full((3, dst_slot_bytes), -1, dtype=torch.int8, device="cuda")

    src_ri = _rank_info(
        name="src", rank=src_rank, tp_size=src_tp, tp_rank=src_rank, heads=src_heads
    )
    dst_ri = _rank_info(
        name="dst", rank=dst_rank, tp_size=dst_tp, tp_rank=dst_rank, heads=dst_heads
    )
    mapper = NHDHeadMismatchMapper(
        transfer_layers=1,
        src_layer_off=0,
        peer_layer_off=0,
        self_ri=src_ri,
        peer_ri=dst_ri,
        self_region_bytes=src_slot_bytes,
        peer_region_bytes=dst_slot_bytes,
        self_pool_num_layers=1,
        peer_pool_num_layers=1,
        self_buffers_per_layer=2,
        peer_buffers_per_layer=2,
    )
    chunk = NHDStagingChunk(
        src_pool=_pool(src),
        src_view=_view(bytes_per_buffer_per_head * src_heads),
        dst_pool=_pool(dst),
        dst_view=_view(bytes_per_buffer_per_head * dst_heads),
        src_block_ids=np.array([2, 0], dtype=np.int64),
        dst_block_ids=np.array([1, 2], dtype=np.int64),
        mapper=mapper,
        staging_offset=0,
    )
    plan = NHDStagingPlan([chunk])
    staging = plan.allocate("cuda")

    plan.pack_into(staging)
    plan.scatter_from(staging)

    src_nhd = src.view(3, 1, 2, 2, src_heads, 2)
    dst_nhd = dst.view(3, 1, 2, 2, dst_heads, 2)
    torch.testing.assert_close(
        dst_nhd[1, :, :, :, dst_head : dst_head + 1, :],
        src_nhd[2, :, :, :, src_head : src_head + 1, :],
    )
    torch.testing.assert_close(
        dst_nhd[2, :, :, :, dst_head : dst_head + 1, :],
        src_nhd[0, :, :, :, src_head : src_head + 1, :],
    )
    assert torch.all(dst_nhd[0] == -1)
    if dst_heads > 1:
        other_heads = [head for head in range(dst_heads) if head != dst_head]
        assert torch.all(dst_nhd[1:3, :, :, :, other_heads, :] == -1)
