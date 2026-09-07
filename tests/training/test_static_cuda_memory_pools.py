from __future__ import annotations

from molt_stream.training.static_cuda_graph import summarize_cuda_memory_pools


def test_cuda_memory_pool_summary_separates_graph_private_segments() -> None:
    result = summarize_cuda_memory_pools([
        {
            "segment_pool_id": (0, 0),
            "total_size": 100,
            "allocated_size": 70,
            "active_size": 80,
        },
        {
            "segment_pool_id": (3, 4),
            "total_size": 200,
            "allocated_size": 120,
            "active_size": 140,
        },
        {
            "segment_pool_id": (3, 4),
            "total_size": 50,
            "allocated_size": 10,
            "active_size": 20,
        },
    ])
    assert result == {
        "default": {
            "reserved_bytes": 100,
            "allocated_bytes": 70,
            "active_bytes": 80,
        },
        "graph_private": {
            "reserved_bytes": 250,
            "allocated_bytes": 130,
            "active_bytes": 160,
        },
    }
