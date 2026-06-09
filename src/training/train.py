import logging
import os
import yaml
import torch
import mlflow
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk


from src.training.model import load_model_and_tokenizer
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

def load_config(path: str = 'configs/train_config.yaml') -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

    
def main():
    setup_logging()
    cfg = load_config()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
          model_name=cfg["model"]["name"],
          lora_cfg=cfg["lora"],
          device=device,
    )

    dataset = load_from_disk('data/processed')

    abs_db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{abs_db_path}")

    experiment_name = cfg["mlflow"]["experiment_name"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=cfg['mlflow']['run_name']):
        mlflow.log_params({
              "model_name": cfg["model"]["name"],
              "device": device,
              "lora_r": cfg["lora"]["r"],
              "lora_alpha": cfg["lora"]["alpha"],
              "lora_dropout": cfg["lora"]["dropout"],
              "num_epochs": cfg["training"]["num_train_epochs"],
              "batch_size": cfg["training"]["per_device_train_batch_size"],
              "learning_rate": cfg["training"]["learning_rate"],
              "max_length": cfg["data"]["max_length"],
        })

        training_args = TrainingArguments(
              output_dir=cfg["training"]["output_dir"],
              num_train_epochs=cfg["training"]["num_train_epochs"],
              per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
              per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
              learning_rate=cfg["training"]["learning_rate"],
              warmup_steps=cfg["training"]["warmup_steps"],
              weight_decay=cfg["training"]["weight_decay"],
              logging_steps=cfg["training"]["logging_steps"],
              eval_strategy=cfg["training"]["eval_strategy"],
              save_strategy=cfg["training"]["save_strategy"],
              load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
              fp16=device == "cuda",
              report_to="mlflow",
          )
        
        trainer = Trainer(
              model=model,
              args=training_args,
              train_dataset=dataset["train"],
              eval_dataset=dataset["test"]
          )
        
        logger.info("Start training...")
        trainer.train()
        logger.info("Training complete.")

        mlflow.log_artifacts(cfg["training"]["output_dir"], artifact_path="model_checkpoint")

if __name__ == "__main__":
    main()

