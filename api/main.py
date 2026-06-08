import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.inference.inference import load_config, load_model_for_inference, ask

app = FastAPI(title="Medical QA API")
app.mount("/static", StaticFiles(directory="api/static"), name="static")

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

@app.get("/")
def index():
    return FileResponse("api/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/ask", response_model=Answer)
def ask_question(body: Question):
    answer = ask(body.question, model, tokenizer, device)
    return Answer(question=body.question, answer=answer, device=device)