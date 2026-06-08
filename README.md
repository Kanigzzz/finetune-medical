# Medical QA Fine-tuning Pipeline

Fine-tuned TinyLlama-1.1B on PubMedQA dataset for medical question answering using LoRA adapters.

## Overview

| Component | Technology |
|---|---|
| Base model | TinyLlama-1.1B-Chat |
| Fine-tuning | LoRA (PEFT) |
| Dataset | PubMedQA (pqa_labeled) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| API | FastAPI |
| Containerization | Docker |

## Results

| Metric | Value |
|---|---|
| eval_loss | 1.393 |
| Training epochs | 3 |
| Trainable parameters | 1,126,400 / 1,101,174,784 (0.10%) |

## Project Structure

```
├── api/                  # FastAPI endpoint
├── configs/              # Training configuration
├── data/                 # Dataset (DVC managed)
├── docker/               # Dockerfile
├── models/               # Saved models
├── notebooks/            # Exploratory analysis
├── src/
│   ├── data/             # Data preparation
│   ├── inference/        # Model inference
│   ├── training/         # Model & training logic
│   └── utils/            # Logging
├── dvc.yaml              # DVC pipeline
├── params.yaml           # Experiment parameters
└── requirements.txt
```

## Quick Start

**Run API locally:**
```bash
git clone https://github.com/Kanigzzz/finetune-medical.git
cd finetune-medical
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Run with Docker:**
```bash
docker build -f docker/Dockerfile -t medical-api .
docker run -p 8000:8000 medical-api
```

**API Usage:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Does aspirin reduce the risk of cardiovascular disease?"}'
```

**Response:**
```json
{
  "question": "Does aspirin reduce the risk of cardiovascular disease?",
  "answer": "yes, aspirin has been shown to reduce...",
  "device": "mps"
}
```

## Training

```bash
# Prepare dataset
python -m src.data.prepare

# Train model
python -m src.training.train

# View experiments
mlflow ui
```

## Model

Fine-tuned adapters available on HuggingFace:
[Kamil123456789/tinyllama-medical](https://huggingface.co/Kamil123456789/tinyllama-medical)
