import torch
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.inference import load_config, load_model_for_inference, ask

app = FastAPI(title="Medical QA API")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


cfg = load_config()

model, tokenizer = load_model_for_inference(cfg['model']['name'],
                                            "Kamil123456789/tinyllama-medical",
                                            device=device)

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    question: str
    answer: str
    device: str

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/ask", response_model=Answer)
def ask_question(body: Question):
    answer = ask(body.question, model, tokenizer, device)
    return Answer(question=body.question, answer=answer, device=device)