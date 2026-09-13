from tools.summarize_cuda_trace import summarize


def test_trace_does_not_count_host_attribution_as_extra_gpu_work():
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "gemm", "dur": 100},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "dur": 110},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaDeviceSynchronize", "dur": 95},
        {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep", "dur": 120},
    ]}
    result = summarize(trace)
    assert result["kernel"]["summed_duration_us"] == 100
    assert result["cuda_runtime"]["summed_duration_us"] == 95
    assert len(result["kernel"]["operations"]) == 1


def test_empty_trace_has_no_observed_kernels():
    assert summarize({"traceEvents": []})["kernel"] == {
        "summed_duration_us": 0, "operations": []
    }
