import torch
import yaml
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_config(path: str = 'configs/train_config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)
    
def load_model_for_inference(base_model_name: str, adapther: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, 
                                                          torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        
    model = PeftModel.from_pretrained(base_model, adapther)
    model.to(device)
    model.eval()

    return model, tokenizer
    
def ask(question: str, model, tokenizer, device: str, max_new_tokens: int = 256) -> str:   
    prompt = (
          f"<instruction>: Answer the question based on the context\n"
          f"<question>: {question}\n"
          f"<answer>:"
      )
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<answer>:")[-1].strip()