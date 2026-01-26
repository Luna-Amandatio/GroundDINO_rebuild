# Compatibility shim: some modules import `groundeddino_vl.datasets` but the
# package directory is `groundeddino_vl/data` in this repo. Re-export the
# transforms module so both import styles work.

from ..data import transforms as transforms

__all__ = ["transforms"]
