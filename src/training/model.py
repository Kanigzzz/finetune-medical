import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig


logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str, lora_cfg: dict, device: str):
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token")

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    lora_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['dropout'],
        task_type=lora_cfg['task_type']
    )

    model = get_peft_model(model, lora_config)

    model.to(device)

    trainable, total = model.get_nb_trainable_parameters()

    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%) ")

    return model, tokenizer