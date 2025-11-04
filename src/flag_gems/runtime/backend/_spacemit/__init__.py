from typing import Any

from backend_utils import VendorInfoBase  # noqa: E402

# from .heuristics_config_utils import HEURISTICS_CONFIGS

from .utils.config_pre_hook import setup_triton_config
setup_triton_config()

vendor_info = VendorInfoBase(
    vendor_name="spacemit", device_name="cpu", device_query_cmd="lscpu"
)

class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = self.idx

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = self.prev_idx
        return False

class _DeviceWrapper:
    def __init__(self, device: Any):
        ...

    def __enter__(self):
        ...

    def __exit__(self, type: Any, value: Any, traceback: Any):
        ...
        return False


CUSTOMIZED_UNUSED_OPS = (
    "contiguous",
    "fill_scalar_",
)


__all__ = ["*"]
