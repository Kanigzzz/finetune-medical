import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Can exercise help reduce blood pressure?"

model.to("mps")
model.eval()

inputs = tokenizer(prompt, return_tensors='pt').to("mps")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))