import pytest

from molt_stream.core.errors import IntegrityError
from molt_stream.training.qlora import _validate_loading_info


def test_clean_loading_report_is_accepted():
    _validate_loading_info({
        "missing_keys": [], "unexpected_keys": [],
        "mismatched_keys": [], "error_msgs": [],
    })


@pytest.mark.parametrize("key", [
    "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
])
def test_partial_base_load_is_not_accepted(key):
    with pytest.raises(IntegrityError, match="Training was refused"):
        _validate_loading_info({key: ["model.layers.0.self_attn.q_norm.weight"]})
