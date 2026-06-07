from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import logging
import yaml


logger = logging.getLogger(__name__)

class PubMedDataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, example):
        
        context = " ".join(example['context']['contexts'])

        format_text = (
            f"<instruction>: By looking to example answer the question\n"
            f"<context>: {context}\n"
            f"<question>: {example['question']}\n"
            f"<answer>: {example['final_decision'] + ' ' + example['long_answer']}\n"
        )

        tokens = self.tokenizer(
            format_text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        tokens['labels'] = tokens['input_ids'].copy()

        return tokens
    

def process_dataset(dataset_name: str, tokenizer: PreTrainedTokenizer, max_length: int=512, test_size: float = 0.1, seed: int = 42):

    logger.info("Loading dataset...")

    dataset = load_dataset(dataset_name, "pqa_labeled")

    split_dataset = dataset['train'].train_test_split(test_size=test_size, seed=seed)
    original_columns = split_dataset['train'].column_names

    processor = PubMedDataProcessor(tokenizer=tokenizer, max_length=512)

    processed_dataset = split_dataset.map(processor,
                                          remove_columns=original_columns,
                                          num_proc=4,
                                          desc="Przetwarzanie danych")
    
    logger.info(f"Train: {len(processed_dataset['train'])}, Test: {len(processed_dataset['test'])}")
    return processed_dataset

if __name__ == "__main__":
    with open("configs/train_config.yaml") as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = process_dataset(
        cfg['data']['dataset_name'],
        tokenizer,
        max_length=cfg["data"]["max_length"],
        test_size=cfg["data"]["test_size"],
        seed=cfg["data"]["seed"],
    )

    dataset.save_to_disk('data/processed')
    logger.info("Dataset Saved")