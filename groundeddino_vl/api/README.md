# GroundedDINO-VL Public API

This module provides the official public API for GroundedDINO-VL - a clean, simple interface for vision-language object detection.

## Quick Start

```python
from groundeddino_vl import load_model, predict

# Load a model
model = load_model("config.py", "weights.pth", device="cuda")

# Run inference
result = predict(model, "photo.jpg", "car . person . dog")

# Use results
print(f"Found {len(result)} objects:")
for label, score in zip(result.labels, result.scores):
    print(f"  {label}: {score:.2f}")
```

## API Overview

### Core Functions

| Function | Description |
|----------|-------------|
| `load_model()` | Load a model from config and checkpoint |
| `predict()` | Run object detection with text prompts |
| `load_image()` | Load and preprocess an image from file |
| `preprocess_image()` | Preprocess an in-memory image |
| `annotate()` | Draw detection results on images |

### Data Structures

| Class | Description |
|-------|-------------|
| `DetectionResult` | Container for detection results with helper methods |

## Features

✅ **Simple**: One-line model loading and inference
✅ **Type-Safe**: Full type hints and runtime validation
✅ **Flexible**: Accepts multiple input formats (paths, numpy, PIL, tensors)
✅ **Structured**: Clean dataclass results instead of tuples
✅ **Well-Documented**: Comprehensive docstrings and examples

## Example Usage

### Basic Detection

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")
result = predict(model, "image.jpg", "car . person")

for label, score in zip(result.labels, result.scores):
    print(f"{label}: {score:.2f}")
```

### Working with OpenCV

```python
import cv2
from groundeddino_vl import load_model, predict, annotate

model = load_model("config.py", "weights.pth")

# Read image with OpenCV
image_bgr = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Detect objects
result = predict(model, image_rgb, "person . car . dog")

# Annotate and save
annotated = annotate(image_rgb, result)
cv2.imwrite("output.jpg", annotated)
```

### Processing Multiple Images

```python
from pathlib import Path
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")

for img_path in Path("images/").glob("*.jpg"):
    result = predict(model, str(img_path), "car . person")
    print(f"{img_path.name}: {len(result)} objects")
```

### Converting Box Formats

```python
from groundeddino_vl import predict

result = predict(model, "image.jpg", "car")

# Normalized cxcywh (default)
print(result.boxes)  # tensor([[0.5, 0.5, 0.2, 0.3]])

# Normalized xyxy
boxes_norm = result.to_xyxy(denormalize=False)
print(boxes_norm)  # tensor([[0.4, 0.35, 0.6, 0.65]])

# Pixel coordinates xyxy
boxes_pixels = result.to_xyxy(denormalize=True)
print(boxes_pixels)  # tensor([[256, 168, 384, 312]])
```

## Design Principles

This API was designed following these principles:

1. **Simplicity First**: Common tasks should be simple
2. **Progressive Disclosure**: Advanced features available but not required
3. **Type Safety**: Full type hints for better IDE support
4. **Sensible Defaults**: Works well out of the box
5. **Clear Errors**: Helpful error messages when things go wrong

## Comparison: Old vs. New

### Old API (GroundingDINO)
```python
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

args = SLConfig.fromfile("config.py")
args.device = "cuda"
model = build_model(args)
checkpoint = torch.load("weights.pth", map_location="cpu")
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model.eval()
# ... many more lines for inference ...
```

### New API (GroundedDINO-VL) ⭐
```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth", device="cuda")
result = predict(model, "image.jpg", "car . person")
```

**Result**: 80% less code!

## Advanced Usage

### Custom Image Preprocessing

```python
from groundeddino_vl import preprocess_image

# Custom preprocessing parameters
tensor = preprocess_image(
    image,
    max_size=1000,  # Custom max size
    size=600        # Custom target size
)
```

### Lower-Level Control

For advanced users who need more control, internal modules are still accessible:

```python
# Advanced: Direct access to internal utilities
from groundeddino_vl.models import build_model
from groundeddino_vl.utils.slconfig import SLConfig
from groundeddino_vl.utils.inference import Model  # Advanced Model class

# Use internal APIs when needed
```

## Documentation

- **API Reference**: See docstrings in [`__init__.py`](./__init__.py)
- **Migration Guide**: See [`MIGRATION_TO_API.md`](../../MIGRATION_TO_API.md)
- **Full Documentation**: See [`API_DESIGN_PHASE.md`](../summaries/API_DESIGN_PHASE.md)

## Testing

Tests are available in [`tests/test_api.py`](../../tests/test_api.py):

```bash
# Run tests
python -m pytest tests/test_api.py -v

# Or use unittest
python tests/test_api.py
```

## Contributing

When adding new functions to the public API:

1. Add comprehensive docstrings with examples
2. Add type hints for all parameters and returns
3. Add tests in `tests/test_api.py`
4. Update this README and main documentation
5. Ensure backward compatibility

## License

Apache License 2.0 - See [LICENSE](../../LICENSE) for details

## Related

- **GroundedDINO-VL**: [Main Repository](https://github.com/ghostcipher1/GroundedDINO-VL)
- **GroundingDINO**: [Original Repository](https://github.com/IDEA-Research/GroundingDINO)
