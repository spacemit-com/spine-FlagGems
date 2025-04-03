from backend_utils import VendorInfoBase  # noqa: E402

from .heuristics_config_utils import HEURISTICS_CONFIGS

global specific_ops, unused_ops
specific_ops = None
unused_ops = None
vendor_info = VendorInfoBase(
    vendor_name="spacemit", device_name="cpu", device_query_cmd="lscpu"
)


def OpLoader():
    global specific_ops, unused_ops
    if specific_ops is None:
        from . import ops  # noqa: F403

        specific_ops = ops.get_specific_ops()
        unused_ops = ops.get_unused_ops()


CUSTOMIZED_UNUSED_OPS = ()


__all__ = ["HEURISTICS_CONFIGS", "vendor_info", "OpLoader"]
