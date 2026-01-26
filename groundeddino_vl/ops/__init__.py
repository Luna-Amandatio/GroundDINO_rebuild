# ------------------------------------------------------------------------
# GroundedDINO-VL - CUDA Operations
# Copyright (c) 2025 GhostCipher. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CUDA/C++ operations for GroundedDINO-VL."""

# Try to import CUDA extension
try:
    from groundeddino_vl import _C

    __cuda_available__ = True
except ImportError:
    _C = None
    __cuda_available__ = False

__all__ = ["_C", "__cuda_available__"]
