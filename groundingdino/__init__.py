"""
GroundingDINO Compatibility Shim

DEPRECATION NOTICE:
This package is now provided by GroundedDINO-VL. The 'groundingdino' namespace
is maintained for backward compatibility only.

For new code, please use:
    import groundeddino_vl
    from groundeddino_vl.models import build_model
    from groundeddino_vl.utils import inference

This compatibility layer will be maintained for the foreseeable future,
but new features will only be added to the groundeddino_vl package.

Original GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
GroundedDINO-VL: https://github.com/ghostcipher1/GroundedDINO-VL

Copyright (c) 2025 GhostCipher. All rights reserved.
Based on GroundingDINO by IDEA-Research (Copyright (c) 2023 IDEA).
Licensed under the Apache License, Version 2.0.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "The 'groundingdino' namespace is deprecated and provided for backward compatibility only. "
    "Please migrate to 'import groundeddino_vl' for new code. "
    "See: https://github.com/ghostcipher1/GroundedDINO-VL for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the new public API for backward compatibility
# Re-export everything from groundeddino_vl
from groundeddino_vl import *  # noqa: F401, F403, E402
from groundeddino_vl import (  # noqa: F401, E402
    DetectionResult,
    __author__,
    __version__,
    annotate,
    load_image,
    load_model,
    predict,
    preprocess_image,
)

# Explicitly re-export commonly used modules for better IDE support
try:
    from groundeddino_vl import models  # noqa: F401
except ImportError:
    pass

try:
    import groundeddino_vl.utils

    # Alias utils to util for backward compatibility
    util = groundeddino_vl.utils
except ImportError:
    pass

try:
    import groundeddino_vl.data

    # Alias data to datasets for backward compatibility
    datasets = groundeddino_vl.data
except ImportError:
    pass

# Provide access to the CUDA extension through this namespace
try:
    from groundeddino_vl import _C  # noqa: F401

    __cuda_available__ = True
except ImportError:
    __cuda_available__ = False

# Build __all__ dynamically based on what's available
__all__ = [
    "__version__",
    "__author__",
    "__cuda_available__",
    # New public API
    "load_model",
    "predict",
    "load_image",
    "preprocess_image",
    "annotate",
    "DetectionResult",
]
if "models" in dir():
    __all__.append("models")
if "util" in dir():
    __all__.extend(["util", "datasets"])
