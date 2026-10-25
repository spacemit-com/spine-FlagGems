"""Shared smt_rvisa eDSL plumbing for the _spacemit vendor ops.

The eDSL only exists on newer spine-triton builds; routing the import through
this module lets every op file keep its regular triton kernel when the module
is missing, the same way ops/__init__ drops mm/flash_attention on old stacks.
"""

import os

from flag_gems.runtime import config_loader

try:
    import triton.language.extra.smt_rvisa as tle
    from triton.language.extra.smt_rvisa import call as _sr_call

    f16 = tle.f16
    f32 = tle.f32
    HAS_SMT_RVISA = True
except ImportError:
    tle = None
    _sr_call = None
    f16 = None
    f32 = None
    HAS_SMT_RVISA = False


def _rvisa_yaml_default() -> bool:
    # default comes from the spacemit_rvisa entry in tune_configs.yaml
    # (vendor_primitive_yaml_config is the raw yaml dict, kept on the loader
    # exactly for vendor-custom keys like this one)
    for cfg in (config_loader.vendor_primitive_yaml_config or {}).get("spacemit_rvisa") or []:
        enabled = (cfg.get("META") or {}).get("enabled")
        if enabled is not None:
            return bool(enabled)
    return False


_RVISA_YAML_DEFAULT = _rvisa_yaml_default()


def smt_rvisa_enabled() -> bool:
    """Opt-in switch for the smt_rvisa eDSL kernels (default off).

    Default lives in tune_configs.yaml (spacemit_rvisa.enabled);
    SPACEMIT_SMT_RVISA=1/0 overrides it per process. The env var is read per
    call so tests can toggle it with monkeypatch.setenv.
    """
    if not HAS_SMT_RVISA:
        return False
    env = os.environ.get("SPACEMIT_SMT_RVISA")
    if env is not None:
        return env == "1"
    return _RVISA_YAML_DEFAULT
