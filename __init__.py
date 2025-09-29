"""lazytorchtools package

Expose the convenience symbols from the single-file module.
"""
from .lazytorchtools import *  # noqa: F401,F403

__all__ = globals().get("__all__", [])
__version__ = "0.1.0"
