# Label Studio Integration

Complete guide for integrating GroundedDINO-VL with Label Studio for automated object detection and annotation.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Workflows](#usage-workflows)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## Overview

GroundedDINO-VL includes an optional **Label Studio ML Backend** that provides:

- **Real-time auto-annotation** using zero-shot object detection
- **FastAPI service** for scalable inference
- **Auto-labeling integration** with Label Studio's "magic wand" feature
- **Batch annotation** support for high-throughput workflows
- **Database logging** with PostgreSQL or SQLite support

### Key Benefits

- **Accelerate labeling**: Auto-generate bounding boxes with AI
- **Reduce manual work**: Focus on reviewing and correcting predictions
- **Flexible prompts**: Use natural language to describe objects
- **Production-ready**: Docker deployment with health checks and monitoring

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (optional but recommended)
- Model weights downloaded (handled automatically)

### 5-Minute Setup

1. **Download model weights**
   ```bash
   mkdir -p models
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth \
     -O models/groundingdino_swinb_cogcoor.pth
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

3. **Access Label Studio**
   - Open browser to `http://localhost:8080`
   - Create admin account on first visit

4. **Connect ML backend**
   - Go to Settings → Machine Learning
   - Add model URL: `http://groundeddino-vl:9090`
   - Click "Validate and Save"

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                                                             │
│  ┌──────────────────┐          ┌────────────────────────┐  │
│  │  Label Studio    │          │  GroundedDINO-VL       │  │
│  │  Port: 8080      │◄────────►│  Port: 9090            │  │
│  │  User: ls        │  ML API  │  Runtime: NVIDIA       │  │
│  │                  │          │  GPU Support           │  │
│  └──────────────────┘          └────────────────────────┘  │
│          │                              │                   │
│          ▼                              ▼                   │
│  ┌──────────────────┐          ┌────────────────────────┐  │
│  │ label-studio-data│          │  models/               │  │
│  │ (Volume)         │          │  groundingdino_swint...│  │
│  └──────────────────┘          └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description | Port |
|-----------|-------------|------|
| **Label Studio** | Annotation interface | 8080 |
| **GroundedDINO-VL Backend** | ML inference service | 9090 |
| **PostgreSQL** (optional) | Prediction history database | 5432 |
| **Data Volume** | Persistent storage | N/A |

---

## Installation

### Method 1: Docker Compose (Recommended)

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  label-studio:
    build:
      context: .
      dockerfile: Dockerfile.labelstudio
    container_name: label-studio
    ports:
      - "8080:8080"
    volumes:
      - ./label-studio-data:/label-studio/data
    environment:
      LABEL_STUDIO_HOST: 0.0.0.0
      LABEL_STUDIO_PORT: 8080
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  groundeddino-vl:
    image: groundeddino-vl:latest
    container_name: groundeddino-vl-server
    ports:
      - "9090:9090"
    volumes:
      - ./models:/app/models
    environment:
      MODEL_CONFIG: /app/models/GroundingDINO_SwinB_cfg.py
      MODEL_CHECKPOINT: /app/models/groundingdino_swinb_cogcoor.pth
      DEVICE: cuda
    runtime: nvidia
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  label-studio-data:
```

**Start services**:
```bash
docker-compose up -d
```

**View logs**:
```bash
docker-compose logs -f
```

**Stop services**:
```bash
docker-compose down
```

### Method 2: Manual Installation

#### Install Label Studio

```bash
# Create virtual environment
python -m venv venv-labelstudio
source venv-labelstudio/bin/activate

# Install Label Studio
pip install label-studio label-studio-sdk

# Start Label Studio
label-studio start --host 0.0.0.0 --port 8080
```

#### Install GroundedDINO-VL Backend

```bash
# Create separate environment
python -m venv venv-backend
source venv-backend/bin/activate

# Install GroundedDINO-VL
pip install groundeddino_vl[ls_backend]

# Start backend server
groundeddino-vl-server \
  --config path/to/config.py \
  --checkpoint path/to/weights.pth \
  --port 9090 \
  --device cuda
```

---

## Configuration

### Backend Configuration

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_CONFIG` | Path to model config file | Required |
| `MODEL_CHECKPOINT` | Path to model weights | Required |
| `DEVICE` | Device for inference | `cuda` |
| `PORT` | Backend service port | `9090` |
| `HOST` | Backend service host | `0.0.0.0` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DB_TYPE` | Database type (`postgresql` or `sqlite`) | `None` |
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_PORT` | PostgreSQL port | `5432` |
| `DB_NAME` | Database name | `groundeddino_predictions` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | Required if using PostgreSQL |

#### Command-Line Arguments

```bash
groundeddino-vl-server --help

# Output:
Usage: groundeddino-vl-server [OPTIONS]

Options:
  --config TEXT           Path to model config file
  --checkpoint TEXT       Path to model checkpoint
  --device TEXT          Device: cuda, cpu, or mps [default: cuda]
  --port INTEGER         Port to run server on [default: 9090]
  --host TEXT            Host to bind to [default: 0.0.0.0]
  --db-type TEXT         Database type: postgresql or sqlite
  --db-host TEXT         PostgreSQL host [default: localhost]
  --db-port INTEGER      PostgreSQL port [default: 5432]
  --db-name TEXT         Database name
  --db-user TEXT         Database user
  --db-password TEXT     Database password
  --help                 Show this message and exit
```

### Label Studio Configuration

#### Add ML Backend

1. Navigate to **Settings** (gear icon)
2. Select **Machine Learning**
3. Click **Add Model**
4. Fill in:
   - **URL**: `http://groundeddino-vl:9090` (Docker) or `http://localhost:9090` (local)
   - **Title**: `GroundedDINO-VL`
   - **Description**: `Zero-shot object detection`
5. Click **Validate and Save**

#### Project Configuration

Create an object detection project with this labeling interface:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="red"/>
    <Label value="car" background="blue"/>
    <Label value="bicycle" background="green"/>
    <Label value="dog" background="yellow"/>
    <Label value="cat" background="orange"/>
    <Label value="truck" background="purple"/>
    <Label value="motorcycle" background="pink"/>
    <Label value="bus" background="brown"/>
  </RectangleLabels>
</View>
```

---

## Usage Workflows

### Workflow 1: Interactive Auto-Annotation

1. **Import images** to Label Studio project
2. **Open a task** in the labeling interface
3. **Click the magic wand icon** (Auto-annotate button)
4. **Review predictions**: Bounding boxes appear automatically
5. **Adjust as needed**: Move, resize, or delete boxes
6. **Submit annotation**: Save corrected labels

### Workflow 2: Batch Prediction

```python
from label_studio_sdk import Client

# Connect to Label Studio
ls = Client(url='http://localhost:8080', api_key='YOUR_API_KEY')

# Get project
project = ls.get_project(PROJECT_ID)

# Trigger batch predictions for all tasks
predictions = project.make_predictions(
    model_version='GroundedDINO-VL'
)

print(f"Generated {len(predictions)} predictions")
```

### Workflow 3: Custom Text Prompts

By default, the backend uses your project's label classes as prompts. To customize:

```python
# In your ML backend code (advanced)
from groundeddino_vl.ls_backend.inference_engine import GroundedDINOInferenceEngine

class CustomInferenceEngine(GroundedDINOInferenceEngine):
    def get_text_prompt(self, task):
        """Override to provide custom prompts."""
        # Extract labels from task schema
        labels = self.get_labels_from_task(task)

        # Add descriptive terms
        enhanced_labels = [f"{label} object" for label in labels]

        return " . ".join(enhanced_labels)
```

### Workflow 4: Export Annotated Data

```bash
# Export annotations from Label Studio
ls export PROJECT_ID \
  --format COCO \
  --output-dir ./annotations/

# Or via UI:
# Project → Export → Select format (COCO, YOLO, Pascal VOC, etc.)
```

---

## Advanced Features

### Database Logging

Track prediction history for analysis and debugging.

#### Using PostgreSQL

```bash
# Start PostgreSQL container
docker run -d \
  --name postgres-labelstudio \
  -e POSTGRES_DB=groundeddino_predictions \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=changeme \
  -p 5432:5432 \
  postgres:15

# Configure backend
groundeddino-vl-server \
  --config config.py \
  --checkpoint weights.pth \
  --db-type postgresql \
  --db-host localhost \
  --db-port 5432 \
  --db-name groundeddino_predictions \
  --db-user postgres \
  --db-password changeme
```

**Database schema**:
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL,
    image_url TEXT NOT NULL,
    text_prompt TEXT NOT NULL,
    predictions JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Using SQLite

```bash
# Configure backend with SQLite
groundeddino-vl-server \
  --config config.py \
  --checkpoint weights.pth \
  --db-type sqlite \
  --db-name predictions.db
```

### Custom Thresholds

Adjust confidence thresholds per project:

```python
# In Label Studio ML backend settings
{
  "model_version": "GroundedDINO-VL",
  "extra_params": {
    "box_threshold": 0.40,
    "text_threshold": 0.30
  }
}
```

### Multi-GPU Support

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 groundeddino-vl-server ...

# Use multiple GPUs (requires custom load balancing)
CUDA_VISIBLE_DEVICES=0,1 groundeddino-vl-server ...
```

---

## Troubleshooting

### Issue: Cannot Connect to ML Backend

**Symptoms:**
- "Connection failed" error
- Red X instead of green checkmark

**Solutions:**

1. **Verify backend is running**
   ```bash
   curl http://localhost:9090/health
   # Expected: {"status":"ok","model_loaded":true}
   ```

2. **Check network connectivity** (Docker)
   ```bash
   docker exec label-studio ping groundeddino-vl
   ```

3. **Review logs**
   ```bash
   docker-compose logs groundeddino-vl
   ```

4. **Restart services**
   ```bash
   docker-compose restart
   ```

### Issue: Predictions Are Empty

**Symptoms:**
- Auto-annotate returns no boxes
- All predictions have zero confidence

**Solutions:**

1. **Lower thresholds**
   ```bash
   # Restart with lower thresholds
   groundeddino-vl-server \
     --config config.py \
     --checkpoint weights.pth \
     --box-threshold 0.25 \
     --text-threshold 0.20
   ```

2. **Check text prompts**
   - Ensure labels match objects in image
   - Use specific, concrete nouns
   - Avoid overly generic terms

3. **Verify model loaded**
   ```bash
   curl http://localhost:9090/health
   ```

### Issue: Slow Inference

**Symptoms:**
- Auto-annotation takes > 10 seconds
- High latency

**Solutions:**

1. **Use GPU acceleration**
   ```bash
   # Ensure CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Resize large images**
   - Label Studio can auto-resize images
   - Configure in Project Settings → Data Manager

3. **Optimize batch size**
   ```python
   # In backend configuration
   INFERENCE_BATCH_SIZE = 4  # Process multiple images
   ```

### Issue: Model Not Loading

**Symptoms:**
- Health check shows `"model_loaded": false`
- Backend fails to start

**Solutions:**

1. **Check weights path**
   ```bash
   ls -lh models/groundingdino_swinb_cogcoor.pth
   ```

2. **Verify config file**
   ```bash
   python -c "from groundeddino_vl.models import build_model; build_model('config.py')"
   ```

3. **Check CUDA compatibility**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

---

## Production Deployment

### Checklist

- [ ] Use PostgreSQL for database (not SQLite)
- [ ] Enable HTTPS with reverse proxy (nginx/Traefik)
- [ ] Set strong authentication passwords
- [ ] Configure resource limits (CPU/memory)
- [ ] Enable monitoring and logging
- [ ] Set up automated backups
- [ ] Use named volumes for data persistence
- [ ] Implement rate limiting
- [ ] Configure firewall rules
- [ ] Set up health checks and alerts

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/labelstudio

upstream labelstudio {
    server 127.0.0.1:8080;
}

upstream groundeddino {
    server 127.0.0.1:9090;
}

server {
    listen 80;
    server_name labelstudio.example.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name labelstudio.example.com;

    ssl_certificate /etc/letsencrypt/live/labelstudio.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/labelstudio.example.com/privkey.pem;

    # Label Studio
    location / {
        proxy_pass http://labelstudio;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # ML Backend
    location /ml/ {
        proxy_pass http://groundeddino/;
        proxy_set_header Host $host;
    }
}
```

### Monitoring

```bash
# Prometheus metrics (add to backend)
pip install prometheus-client

# In server.py
from prometheus_client import Counter, Histogram, generate_latest

PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Reference Links

- **Label Studio Docs**: [labelstud.io/guide](https://labelstud.io/guide/)
- **ML Backend Guide**: [labelstud.io/guide/ml.html](https://labelstud.io/guide/ml.html)
- **Docker Compose**: [docs.docker.com/compose](https://docs.docker.com/compose/)
- **GroundedDINO-VL API**: [API_REFERENCE.md](API_REFERENCE.md)

---

**Need Help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on [GitHub](https://github.com/ghostcipher1/GroundedDINO-VL/issues).
