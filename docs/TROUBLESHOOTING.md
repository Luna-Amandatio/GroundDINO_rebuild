# Troubleshooting Guide

Common issues and solutions for GroundedDINO-VL.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Import and Module Errors](#import-and-module-errors)
- [CUDA and GPU Issues](#cuda-and-gpu-issues)
- [Model Loading Issues](#model-loading-issues)
- [Inference and Prediction Issues](#inference-and-prediction-issues)
- [Performance Issues](#performance-issues)
- [Label Studio Integration Issues](#label-studio-integration-issues)
- [Build and Compilation Issues](#build-and-compilation-issues)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: "ERROR: Could not find a version that satisfies the requirement"

**Cause:** Package not available for your Python version or platform

**Solution:**
```bash
# Check Python version (must be 3.9-3.12)
python --version

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Try installing again
pip install groundeddino_vl
```

### Issue: "No matching distribution found for groundeddino_vl"

**Cause:** PyPI connectivity issues or typo in package name

**Solution:**
```bash
# Verify package name (note: groundeddino_vl with underscore)
pip install groundeddino_vl

# If PyPI is down, try a mirror
pip install groundeddino_vl -i https://mirrors.aliyun.com/pypi/simple/

# Or install from source
pip install git+https://github.com/ghostcipher1/GroundedDINO-VL.git
```

### Issue: "permission denied" during installation

**Cause:** Insufficient permissions to write to system directories

**Solution:**
```bash
# Use --user flag (installs to user directory)
pip install --user groundeddino_vl

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install groundeddino_vl
```

---

## Import and Module Errors

### Issue: "ModuleNotFoundError: No module named 'groundeddino_vl'"

**Cause:** Package not installed in current environment

**Solution:**
```bash
# Verify which Python is being used
which python
python --version

# Check installed packages
pip list | grep groundeddino

# Reinstall if needed
pip install groundeddino_vl

# Verify installation
python -c "import groundeddino_vl; print(groundeddino_vl.__version__)"
```

### Issue: "ImportError: cannot import name '_C'"

**Cause:** CUDA extensions not built or incompatible

**Solution:**
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CPU version or fix CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# If True but still fails, rebuild extensions
pip install --force-reinstall --no-cache-dir groundeddino_vl -v
```

### Issue: "AttributeError: module 'groundeddino_vl' has no attribute 'load_model'"

**Cause:** Using old API or incorrect import

**Solution:**
```python
# Correct import (new API)
from groundeddino_vl import load_model, predict

# If using legacy API
from groundeddino_vl.utils.inference import Model

# Verify API availability
import groundeddino_vl
print(dir(groundeddino_vl))  # Should show: load_model, predict, annotate
```

---

## CUDA and GPU Issues

### Issue: "RuntimeError: CUDA out of memory"

**Cause:** GPU memory exhausted

**Solution:**
```python
# Solution 1: Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Solution 2: Use smaller images
from PIL import Image
image = Image.open("large_image.jpg")
image = image.resize((1280, 720))  # Resize before inference

# Solution 3: Use mixed precision
with torch.cuda.amp.autocast():
    result = predict(model, image, text_prompt)

# Solution 4: Switch to CPU
model = load_model("config.py", "weights.pth", device="cpu")
```

### Issue: "CUDA error: device-side assert triggered"

**Cause:** Invalid tensor operations or CUDA version mismatch

**Solution:**
```bash
# Check CUDA and PyTorch versions match
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall compatible PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Enable CUDA error checking for debugging
export CUDA_LAUNCH_BLOCKING=1
python your_script.py
```

### Issue: "torch.cuda.is_available() returns False"

**Cause:** CUDA not detected or driver issue

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install NVIDIA driver
# Ubuntu/Debian:
sudo apt-get install nvidia-driver-550

# Check CUDA installation
ls /usr/local/cuda*/

# Set CUDA_HOME (Linux)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify PyTorch CUDA
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

## Model Loading Issues

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'config.py'"

**Cause:** Config or checkpoint file not found

**Solution:**
```python
# Use absolute paths
from pathlib import Path

config_path = Path("/full/path/to/config.py").resolve()
checkpoint_path = Path("/full/path/to/weights.pth").resolve()

model = load_model(str(config_path), str(checkpoint_path))

# Or download automatically
python -m groundeddino_vl download-weights
```

### Issue: "RuntimeError: Error(s) in loading state_dict"

**Cause:** Checkpoint incompatible with model architecture

**Solution:**
```python
# Verify checkpoint integrity
import torch
checkpoint = torch.load("weights.pth", map_location="cpu")
print(checkpoint.keys())  # Should show: model, optimizer, etc.

# Try loading with strict=False
model = load_model("config.py", "weights.pth", strict=False)

# Download official weights
python -m groundeddino_vl download-weights
```

### Issue: Model weights fail to download automatically

**Cause:** Network connectivity or HuggingFace Hub issues

**Solution:**
```bash
# Manual download
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth \
  -O ~/.cache/groundeddino-vl/groundingdino_swint_ogc.pth

# Or use Python
from groundeddino_vl.weights_manager import download_model_weights
weights_path = download_model_weights(force=True)
print(f"Downloaded to: {weights_path}")

# Set custom cache directory
export GDVL_CACHE_DIR=/custom/cache/dir
python -m groundeddino_vl download-weights
```

---

## Inference and Prediction Issues

### Issue: predict() returns empty results

**Cause:** Thresholds too high or objects not matching prompts

**Solution:**
```python
# Lower thresholds
result = predict(
    model=model,
    image="photo.jpg",
    text_prompt="person . car",
    box_threshold=0.25,  # Lower from default 0.35
    text_threshold=0.20,  # Lower from default 0.25
)

# Try different prompts
# Instead of generic terms:
result = predict(model, "photo.jpg", "vehicle")  # Too generic

# Use specific objects:
result = predict(model, "photo.jpg", "car . truck . bus")  # Better

# Check if objects are actually in the image
# Visualize to debug
from groundeddino_vl import annotate
annotated = annotate(image, result)
```

### Issue: Predictions have very low confidence scores

**Cause:** Poor text-image alignment or difficult objects

**Solution:**
```python
# Use more descriptive prompts
# Bad:
result = predict(model, image, "thing")

# Better:
result = predict(model, image, "red car . person wearing blue shirt")

# Try synonyms
result = predict(model, image, "automobile . vehicle . sedan")

# Filter low-confidence results
high_conf_results = [
    (label, score, box)
    for label, score, box in zip(result.labels, result.scores, result.boxes)
    if score >= 0.40
]
```

### Issue: "ValueError: Image format not supported"

**Cause:** Invalid image input type

**Solution:**
```python
# Ensure proper image format
import cv2
import numpy as np
from PIL import Image

# From file path (str)
result = predict(model, "image.jpg", "person")

# From numpy array (RGB, uint8)
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = predict(model, image_rgb, "person")

# From PIL Image
pil_image = Image.open("image.jpg").convert("RGB")
result = predict(model, pil_image, "person")

# Check array shape and dtype
print(f"Shape: {image_rgb.shape}")  # Should be (H, W, 3)
print(f"Dtype: {image_rgb.dtype}")  # Should be uint8
```

---

## Performance Issues

### Issue: Slow inference speed (> 5 seconds per image)

**Cause:** CPU inference or large images

**Solution:**
```python
# 1. Use GPU
model = load_model("config.py", "weights.pth", device="cuda")

# 2. Resize large images
from PIL import Image
image = Image.open("huge_image.jpg")
image = image.resize((1280, 720), Image.LANCZOS)

# 3. Use mixed precision (GPU only)
import torch
with torch.cuda.amp.autocast():
    result = predict(model, image, "person")

# 4. Batch processing (if using low-level API)
from groundeddino_vl.utils.inference import Model
model = Model("config.py", "weights.pth")
# Process multiple images without reloading model
```

### Issue: High memory usage

**Cause:** Memory leaks or large batch sizes

**Solution:**
```python
import torch

# Clear cache periodically
for i, image in enumerate(images):
    result = predict(model, image, "person")

    if i % 100 == 0:
        torch.cuda.empty_cache()

# Delete references
del result
del image

# Use context manager for inference
with torch.no_grad():
    result = predict(model, image, "person")
```

### Issue: Model loading takes too long

**Cause:** Loading from slow storage or network

**Solution:**
```bash
# Move weights to faster storage (SSD)
mv weights.pth /path/to/ssd/weights.pth

# Or cache in RAM (Linux)
sudo mount -t tmpfs -o size=2G tmpfs /tmp/model_cache
cp weights.pth /tmp/model_cache/

# Load once and reuse
model = load_model("config.py", "weights.pth")
# Reuse 'model' for all predictions
```

---

## Label Studio Integration Issues

See the complete [Label Studio Integration Guide](LABEL_STUDIO.md) for detailed troubleshooting.

### Quick Fixes

**Issue**: Cannot connect ML backend to Label Studio

**Solution**:
```bash
# 1. Verify backend is running
curl http://localhost:9090/health

# 2. Check Docker network (if using Docker)
docker network inspect groundeddino-vl_default

# 3. Use correct URL
# Docker: http://groundeddino-vl:9090
# Local: http://localhost:9090
```

**Issue**: Auto-annotation returns no boxes

**Solution**:
```bash
# Restart backend with lower thresholds
groundeddino-vl-server \
  --config config.py \
  --checkpoint weights.pth \
  --box-threshold 0.25 \
  --text-threshold 0.20
```

---

## Build and Compilation Issues

### Issue: "error: invalid argument '-std=c++17' not allowed with 'C'"

**Cause:** Trying to compile C files with C++ flags

**Solution:**
```bash
# Ensure all CUDA files use .cpp extension
find . -name "*.c" -type f

# Rename if needed
for file in $(find . -name "*.c"); do
    mv "$file" "${file%.c}.cpp"
done

# Rebuild
pip install --force-reinstall --no-cache-dir .
```

### Issue: "CUDA_HOME is not set"

**Cause:** CUDA Toolkit not detected

**Solution:**
```bash
# Find CUDA installation
ls /usr/local/cuda*/

# Set CUDA_HOME (Linux/macOS)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Windows (PowerShell)
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"

# Make permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
source ~/.bashrc
```

### Issue: Compilation fails on Windows

**Cause:** Missing Visual Studio C++ tools

**Solution:**
1. Install Visual Studio 2019 or newer
2. During installation, select:
   - "Desktop development with C++"
   - "MSVC v142+ build tools"
   - "Windows 10/11 SDK"
3. Restart computer
4. Try building again

---

## Getting Help

### Before Asking for Help

1. **Check this guide** for your specific issue
2. **Review documentation**:
   - [Installation Guide](INSTALLATION.md)
   - [Quick Start](QUICKSTART.md)
   - [API Reference](API_REFERENCE.md)
3. **Search existing issues** on [GitHub](https://github.com/ghostcipher1/GroundedDINO-VL/issues)
4. **Gather diagnostic information**:
   ```bash
   # System info
   python --version
   pip list | grep -E "(torch|groundeddino)"
   nvidia-smi  # If using GPU

   # Test imports
   python -c "import groundeddino_vl; print(groundeddino_vl.__version__)"
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### How to Report Issues

When creating a GitHub issue, include:

1. **Environment**:
   - OS and version
   - Python version
   - GroundedDINO-VL version
   - PyTorch version
   - CUDA version (if using GPU)

2. **Problem description**:
   - What you're trying to do
   - What you expected to happen
   - What actually happened

3. **Code to reproduce**:
   ```python
   # Minimal reproducible example
   from groundeddino_vl import load_model, predict

   model = load_model("config.py", "weights.pth")
   result = predict(model, "image.jpg", "person")
   # Error occurs here
   ```

4. **Error message**:
   ```
   # Full traceback
   Traceback (most recent call last):
     ...
   ```

5. **What you've tried**:
   - Steps taken to resolve
   - Related issues or documentation reviewed

### Community Support

- **GitHub Issues**: [github.com/ghostcipher1/GroundedDINO-VL/issues](https://github.com/ghostcipher1/GroundedDINO-VL/issues)
- **Discussions**: [github.com/ghostcipher1/GroundedDINO-VL/discussions](https://github.com/ghostcipher1/GroundedDINO-VL/discussions)

---

## Diagnostic Commands

Quick reference for troubleshooting:

```bash
# Python and packages
python --version
pip list | grep -E "(torch|groundeddino|numpy|PIL)"

# CUDA and GPU
nvidia-smi
nvcc --version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# GroundedDINO-VL
python -c "import groundeddino_vl; print(f'Version: {groundeddino_vl.__version__}'); print(f'CUDA available: {groundeddino_vl.__cuda_available__}')"

# Test inference
python -c "
from groundeddino_vl import load_model, predict
import numpy as np
model = load_model('config.py', 'weights.pth', device='cpu')
fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = predict(model, fake_image, 'person . car')
print(f'Inference successful: {len(result)} detections')
"

# Disk space
df -h
du -sh ~/.cache/groundeddino-vl/

# Process and memory
ps aux | grep python
free -h
```

---

**Still stuck?** Open an issue with diagnostic output and detailed description: [GitHub Issues](https://github.com/ghostcipher1/GroundedDINO-VL/issues)
