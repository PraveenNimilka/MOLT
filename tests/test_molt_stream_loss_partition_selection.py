from molt_stream.training.loss_partition import select_partition_candidate


def test_partition_selector_rejects_inexact_fast_result():
    result = select_partition_candidate([
        {
            "chunk_size": 128,
            "gradient_close": False,
            "loss_close": True,
            "step_speedup_percent": 50.0,
            "peak_memory_reduction_percent": 50.0,
        }
    ])
    assert result["survived"] is False


def test_partition_selector_accepts_memory_pareto_result():
    result = select_partition_candidate([
        {
            "chunk_size": 2048,
            "gradient_close": True,
            "loss_close": True,
            "step_speedup_percent": -5.0,
            "peak_memory_reduction_percent": 20.0,
        }
    ])
    assert result["survived"] is True
    assert result["selected_chunk_size"] == 2048
