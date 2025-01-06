from glob import glob
import logging
import os
import sys
import time
from typing import List

import oci
from oci.object_storage import ObjectStorageClient
import ocifs

from peft import get_peft_model

import torch
from torch.utils.data import IterableDataset, DistributedSampler, DataLoader
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.nn.utils.rnn import pad_sequence

from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, Trainer
from transformers import set_seed

from datasets import load_dataset

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model

#-------------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ocifs_mount')
logger.setLevel(logging.INFO)

class C4IterableDataset(IterableDataset):
    def __init__(self, dataset_path: str, num_samples: int, tokenizer: AutoTokenizer, world_size:int, rank: int):
        self.all_datafiles = sorted(glob(os.path.join(dataset_path, 'c4-train*.json.gz')))
        time.sleep(5)
        self.datafiles = self.all_datafiles[rank::world_size]
        self.num_samples = num_samples
        self.tokenizer = tokenizer
    
    def __iter__(self):
        dataset = load_dataset('json', data_files={'train': self.datafiles}, streaming=True)
        dataset_iter = iter(dataset['train'])
        count = 0
        for sample in dataset_iter:
            if count >= self.num_samples:
                break
            tokenized_samples = self.tokenizer(
                sample['text'],
                truncation = True,
                padding='max_length',
                max_length=2048,
                return_tensors="pt"
            )
            yield {
                "input_ids": tokenized_samples["input_ids"].squeeze(0),
                "attention_mask": tokenized_samples["attention_mask"].squeeze(0)
            }
            count += 1

    def __len__(self):
        return self.num_samples

#-------------------------------------------------------------------------------
def collate_fn(batch):
    # Extract input_ids and attention_mask from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['input_ids'] for item in batch]
    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored tokens

    # Return a dictionary of tensors
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

#-------------------------------------------------------------------------------
def main(model_args: ModelArguments,
         data_args: DataTrainingArguments,
         training_args: TrainingArguments,
         ddp_args: DDPArguments):

    if not is_initialized():
        print("Distributed process not initialized.")
        return

    rank = get_rank()
    world_size = get_world_size()
    print(f"Rank {rank}/{world_size} initialized successfully on node {os.environ.get('NODE_RANK', 'unknown')}.")
    print(f"{os.environ['WORLD_SIZE']=}")
    print(f"{os.environ['LOCAL_WORLD_SIZE']=}")
    print(f"{os.environ['RANK']=}")
    print(f"{os.environ['MASTER_ADDR']=}")
    print(f"{os.environ['MASTER_PORT']=}")
    print(f"{os.environ['LOCAL_RANK']=}")
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained("/nfs/cluster/models/meta-llama/Llama-3.1-70B/")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = C4IterableDataset(dataset_path='/nfs/cluster/allenai_c4_en_raw/',
                                num_samples=22_804_306,
                                tokenizer=tokenizer,
                                world_size=world_size,
                                rank=rank)
    
    model, _, peft_config = create_and_prepare_model(model_args)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = not training_args.gradient_checkpointing
    model.config.return_dict = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DDPArguments)
    )
    model_args, data_args, training_args, ddp_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.local_cache_path, exist_ok=True)
    logging.info(f"Created {data_args.local_cache_path}")
    main(model_args, data_args, training_args, ddp_args)