# Quick Start Guide

This guide provides practical examples to get you started with GroundedDINO-VL for zero-shot object detection tasks.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Basic Usage](#basic-usage)
  - [Modern High-Level API](#modern-high-level-api)
  - [Detection from Image Paths](#detection-from-image-paths)
  - [Detection from Image Arrays](#detection-from-image-arrays)
  - [Batch Processing](#batch-processing)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
  - [Low-Level API with Supervision](#low-level-api-with-supervision)
  - [Custom Thresholds](#custom-thresholds)
  - [Class-Based Detection](#class-based-detection)
- [Common Use Cases](#common-use-cases)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Prerequisites

Ensure GroundedDINO-VL is installed:

```bash
pip install groundeddino_vl
```

Download model weights (handled automatically on first use):

```bash
# Optional: Pre-download weights explicitly
python -m groundeddino_vl download-weights
```

---

## Basic Usage

### Modern High-Level API

The recommended way to use GroundedDINO-VL is through the clean, high-level API:

```python
from groundeddino_vl import load_model, predict

# Load model once (auto-downloads weights if needed)
model = load_model(
    config_path="path/to/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="path/to/groundingdino_swint_ogc.pth",
    device="cuda"  # or "cpu"
)

# Run detection with text prompt
result = predict(
    model=model,
    image="path/to/image.jpg",
    text_prompt="car . person . dog",  # Objects separated by " . "
    box_threshold=0.35,  # Confidence threshold for boxes
    text_threshold=0.25,  # Confidence threshold for text matching
)

# Access results
print(f"Found {len(result)} objects")
for label, score, box in zip(result.labels, result.scores, result.boxes):
    print(f"{label}: {score:.2f} at {box}")
```

**Result Object Structure:**
```python
result.labels       # List of detected object labels
result.scores       # List of confidence scores (0-1)
result.boxes        # List of bounding boxes in [cx, cy, w, h] format (normalized)
result.image_size   # Tuple of (height, width)
```

---

### Detection from Image Paths

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth", device="cuda")

# Detect objects in an image file
result = predict(
    model=model,
    image="photo.jpg",
    text_prompt="person . bicycle . car . motorcycle . traffic light",
    box_threshold=0.35,
    text_threshold=0.25,
)

# Convert boxes to pixel coordinates (xyxy format)
boxes_xyxy = result.to_xyxy(denormalize=True)
print(f"Boxes (xyxy): {boxes_xyxy}")

# Or get boxes in xywh format
boxes_xywh = result.to_xywh(denormalize=True)
print(f"Boxes (xywh): {boxes_xywh}")
```

---

### Detection from Image Arrays

```python
import cv2
from groundeddino_vl import load_model, predict

# Load image with OpenCV (BGR format)
image_bgr = cv2.imread("photo.jpg")

# Convert BGR to RGB (GroundedDINO-VL expects RGB)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

model = load_model("config.py", "weights.pth")

# Predict on numpy array
result = predict(
    model=model,
    image=image_rgb,  # Pass RGB numpy array
    text_prompt="cat . dog . bird",
)

print(f"Detected {len(result)} objects: {result.labels}")
```

**Supported Image Formats:**
- File paths (str): `"image.jpg"`, `"path/to/photo.png"`
- NumPy arrays (RGB): `numpy.ndarray` with shape `(H, W, 3)`
- PIL Images: `PIL.Image.Image` objects
- PyTorch tensors: `torch.Tensor` with shape `(3, H, W)`

---

### Batch Processing

Process multiple images efficiently:

```python
from pathlib import Path
from groundeddino_vl import load_model, predict

# Load model once
model = load_model("config.py", "weights.pth")

# Get all images in directory
image_dir = Path("images/")
image_paths = list(image_dir.glob("*.jpg"))

# Process each image
results = []
for image_path in image_paths:
    result = predict(
        model=model,
        image=str(image_path),
        text_prompt="person . vehicle . animal",
        box_threshold=0.35,
    )
    results.append({
        "image": image_path.name,
        "detections": len(result),
        "labels": result.labels,
        "scores": result.scores,
    })

# Save results to JSON
import json
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

---

## Visualization

### Basic Annotation

```python
from groundeddino_vl import load_model, predict, annotate
import cv2

model = load_model("config.py", "weights.pth")

# Load image (returns RGB numpy array)
image_rgb = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

# Run detection
result = predict(
    model=model,
    image=image_rgb,
    text_prompt="person . car . bicycle",
)

# Annotate image (returns BGR format for OpenCV)
annotated = annotate(
    image=image_rgb,
    result=result,
    show_labels=True,
    show_confidence=True,
)

# Save or display
cv2.imwrite("output.jpg", annotated)
cv2.imshow("Detection Result", annotated)
cv2.waitKey(0)
```

### Custom Visualization with Matplotlib

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")
result = predict(model, "photo.jpg", "cat . dog")

# Get denormalized boxes in xyxy format
boxes = result.to_xyxy(denormalize=True)

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(plt.imread("photo.jpg"))

# Draw bounding boxes
for box, label, score in zip(boxes, result.labels, result.scores):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1

    # Draw rectangle
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # Add label
    ax.text(
        x1, y1 - 5,
        f"{label}: {score:.2f}",
        color='white',
        fontsize=12,
        bbox=dict(facecolor='red', alpha=0.7)
    )

plt.axis('off')
plt.tight_layout()
plt.savefig("annotated_output.jpg", dpi=150, bbox_inches='tight')
plt.show()
```

---

## Advanced Usage

### Low-Level API with Supervision

For advanced users who need fine-grained control:

```python
import cv2
from groundeddino_vl.utils.inference import Model
import supervision as sv

# Load image (BGR format)
image_bgr = cv2.imread("photo.jpg")

# Initialize model with low-level API
model = Model(
    model_config_path="config.py",
    model_checkpoint_path="weights.pth"
)

# Predict with caption (returns sv.Detections + labels)
detections, labels = model.predict_with_caption(
    image=image_bgr,
    caption="person . car . bicycle",
    box_threshold=0.35,
    text_threshold=0.25,
)

# Visualize with Supervision
box_annotator = sv.BoxAnnotator()
annotated = box_annotator.annotate(scene=image_bgr, detections=detections)

# Add labels with confidence
label_annotator = sv.LabelAnnotator()
labels_with_conf = [
    f"{label} {conf:.2f}"
    for label, conf in zip(labels, detections.confidence)
]
annotated = label_annotator.annotate(
    scene=annotated,
    detections=detections,
    labels=labels_with_conf
)

cv2.imshow("Result", annotated)
cv2.waitKey(0)
```

### Custom Thresholds

Adjust detection sensitivity:

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")

# High precision (fewer false positives)
result_strict = predict(
    model=model,
    image="photo.jpg",
    text_prompt="person",
    box_threshold=0.50,  # Higher threshold
    text_threshold=0.35,
)

# High recall (fewer false negatives)
result_lenient = predict(
    model=model,
    image="photo.jpg",
    text_prompt="person",
    box_threshold=0.25,  # Lower threshold
    text_threshold=0.20,
)

print(f"Strict detections: {len(result_strict)}")
print(f"Lenient detections: {len(result_lenient)}")
```

**Threshold Guidelines:**
- `box_threshold`: Confidence for bounding box (0.0 - 1.0)
  - **0.25-0.30**: High recall, more false positives
  - **0.35-0.40**: Balanced (recommended)
  - **0.50+**: High precision, may miss objects
- `text_threshold`: Confidence for text-image alignment (0.0 - 1.0)
  - **0.20**: More lenient text matching
  - **0.25**: Balanced (recommended)
  - **0.30+**: Stricter text matching

### Class-Based Detection

Detect specific predefined classes:

```python
from groundeddino_vl.utils.inference import Model

model = Model("config.py", "weights.pth")

# Define classes of interest
classes = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]

# Predict with class IDs
detections = model.predict_with_classes(
    image="photo.jpg",
    classes=classes,
    box_threshold=0.35,
    text_threshold=0.25,
)

# class_id field is automatically populated
for i in range(len(detections)):
    class_id = detections.class_id[i]
    confidence = detections.confidence[i]
    class_name = classes[class_id]
    print(f"Detected {class_name} with confidence {confidence:.2f}")
```

---

## Common Use Cases

### Use Case 1: Traffic Monitoring

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")

# Detect vehicles and traffic elements
result = predict(
    model=model,
    image="traffic_scene.jpg",
    text_prompt="car . truck . bus . motorcycle . traffic light . stop sign",
    box_threshold=0.35,
)

# Count vehicles by type
vehicle_counts = {}
for label in result.labels:
    vehicle_counts[label] = vehicle_counts.get(label, 0) + 1

print("Traffic summary:", vehicle_counts)
```

### Use Case 2: Retail Shelf Analysis

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")

# Detect products on shelf
result = predict(
    model=model,
    image="retail_shelf.jpg",
    text_prompt="bottle . can . box . package",
    box_threshold=0.30,
)

print(f"Total products detected: {len(result)}")
print(f"Product breakdown: {dict(zip(*np.unique(result.labels, return_counts=True)))}")
```

### Use Case 3: Wildlife Monitoring

```python
from groundeddino_vl import load_model, predict
from pathlib import Path

model = load_model("config.py", "weights.pth")

# Process trail camera images
camera_images = Path("trail_cam/").glob("*.jpg")

wildlife_log = []
for image_path in camera_images:
    result = predict(
        model=model,
        image=str(image_path),
        text_prompt="deer . bear . fox . raccoon . bird",
        box_threshold=0.40,  # Higher threshold for wildlife
    )

    if len(result) > 0:
        wildlife_log.append({
            "timestamp": image_path.stem,
            "animals": result.labels,
            "count": len(result),
        })

print(f"Wildlife detected in {len(wildlife_log)} images")
```

### Use Case 4: Safety Equipment Detection

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")

# Detect safety equipment on construction site
result = predict(
    model=model,
    image="construction_site.jpg",
    text_prompt="hard hat . safety vest . person . gloves . safety goggles",
    box_threshold=0.35,
)

# Check compliance
persons = sum(1 for label in result.labels if label == "person")
hard_hats = sum(1 for label in result.labels if label == "hard hat")
safety_vests = sum(1 for label in result.labels if label == "safety vest")

print(f"Persons detected: {persons}")
print(f"Hard hats detected: {hard_hats}")
print(f"Safety vests detected: {safety_vests}")
print(f"Compliance: {hard_hats >= persons and safety_vests >= persons}")
```

---

## Tips and Best Practices

### 1. Prompt Engineering

**Good Prompts:**
- Use specific, concrete nouns: `"car"`, `"person"`, `"dog"`
- Separate with ` . ` (space-dot-space): `"cat . dog . bird"`
- Use singular forms: `"person"` not `"people"`
- Be descriptive: `"red car"`, `"sitting dog"`

**Avoid:**
- Vague terms: `"thing"`, `"object"`
- Verbs alone: `"running"`, `"jumping"`
- Complex phrases: `"a person wearing a red hat"`

### 2. Performance Optimization

```python
# Reuse model across predictions
model = load_model("config.py", "weights.pth")

# Use appropriate batch sizes
# Process multiple images in sequence without reloading model

# Enable mixed precision for faster inference (GPU)
import torch
with torch.cuda.amp.autocast():
    result = predict(model, image, text_prompt)

# Use smaller input sizes for faster processing
# Images are automatically resized while maintaining aspect ratio
```

### 3. Memory Management

```python
import torch
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth", device="cuda")

# Process images and clear cache periodically
for i, image_path in enumerate(image_paths):
    result = predict(model, image_path, "person . car")

    # Clear CUDA cache every 100 images
    if i % 100 == 0:
        torch.cuda.empty_cache()
```

### 4. Error Handling

```python
from groundeddino_vl import load_model, predict

try:
    model = load_model("config.py", "weights.pth")
except FileNotFoundError:
    print("Model weights not found. Downloading...")
    # Weights will auto-download

try:
    result = predict(model, "photo.jpg", "person . car")
except Exception as e:
    print(f"Prediction failed: {e}")
    # Handle error appropriately
```

### 5. Confidence Filtering

```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")
result = predict(model, "photo.jpg", "person . car . dog")

# Filter low-confidence detections
high_confidence_results = [
    (label, score, box)
    for label, score, box in zip(result.labels, result.scores, result.boxes)
    if score >= 0.5
]

print(f"High-confidence detections: {len(high_confidence_results)}")
```

---

## Next Steps

- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation
- **Label Studio Integration**: Check [LABEL_STUDIO.md](LABEL_STUDIO.md) for auto-annotation setup
- **Advanced Topics**: Review [Building from Source](../BUILD_GUIDE.md) for custom builds

---

**Need Help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/ghostcipher1/GroundedDINO-VL/issues).
