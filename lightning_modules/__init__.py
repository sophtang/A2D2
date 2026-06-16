from .mdm import MaskedDiffusionModule
from .any_order import AnyOrderInsertionFlowModule


__all__ = [
    "MaskedDiffusionModule",
    "AutoregressiveModule",
    "AnyOrderInsertionFlowModule",
]


def __getattr__(name):
    if name == "AutoregressiveModule":
        from .autoregressive import AutoregressiveModule
        return AutoregressiveModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
