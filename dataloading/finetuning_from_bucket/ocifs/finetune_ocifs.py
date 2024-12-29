import gzip
import json
import logging
import os
import sys
from typing import List

import oci
from oci.object_storage import ObjectStorageClient
import ocifs

import torch
from torch.utils.data import IterableDataset, DistributedSampler
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, Trainer
from transformers import set_seed

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model

#-------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ocifs')
logger.setLevel(logging.INFO)


#-------------------------------------------------------------------------------
class OciFsFileReader:
    def __init__(self,
                 file_path: str,
                 fs: ocifs.OCIFileSystem,
                 tokenizer: AutoTokenizer,
                 col_to_use: str = "text",
                 max_len: int = 2048,
                 truncation: bool = True):
        self.file_path = file_path
        self.fs = fs
        self.tokenizer = tokenizer
        self.col_to_use = col_to_use
        self.max_len = max_len
        self.truncation = truncation
        self.samples = self._load_file()

    #---------------------------------------------------------------------------
    def _load_file(self):
        # Read a single file content from OCI Object storage
        with self.fs.open(self.file_path, "rb") as f:
            with gzip.GzipFile(fileobj=f) as gz:
                lines = gz.readlines()
        data = [json.loads(line.decode("utf-8")) for line in lines]
        return data

    #---------------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    #---------------------------------------------------------------------------
    def __getitem__(self, idx):
        return self.samples[idx]

#-------------------------------------------------------------------------------
class C4OciIterableDataset(IterableDataset):
    def __init__(self,
                 file_list: List[str],
                 fs: ocifs.OCIFileSystem,
                 tokenizer: AutoTokenizer = None,
                 col_to_use: str = "text",
                 max_len: int = 2048,
                 truncation: bool = True):
        """
        Args:
            file_list (list): List of file paths in the bucket.
            fs (OCIFileSystem): Initialized OCI filesystem object.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.file_list = file_list
        self.fs = fs
        self.tokenizer = tokenizer
        self.col_to_use = col_to_use
        self.max_len = max_len
        self.truncation = truncation

    #---------------------------------------------------------------------------
    def _partition_files(self):
        """Assign a subset of files to this worker."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.file_list
        else:
            # Partition files across workers
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            return self.file_list[worker_id::total_workers]

    #---------------------------------------------------------------------------
    def _read_files(self):
        partitioned_files = self._partition_files()
        for file_path in partitioned_files:
            reader = OciFsFileReader(
                file_path=file_path,
                fs=self.fs,
                tokenizer=self.tokenizer,
                col_to_use=self.col_to_use,
                max_len=self.max_len,
                truncation=self.truncation
            )
            for sample in reader:
                # Tokenize the current sample and return tensors
                tokenized_sample = self.tokenizer(
                    sample[self.col_to_use],
                    truncation=self.truncation,
                    padding="max_length",
                    max_length=self.max_len,
                    return_tensors="pt",  # Ensure PyTorch tensors are returned
                )
                # tokenized_sample is a dict with tensors. Yield as-is.
                yield {
                    "input_ids": tokenized_sample["input_ids"].squeeze(0),
                    "attention_mask": tokenized_sample["attention_mask"].squeeze(0),
                }

    #---------------------------------------------------------------------------                                
    def __iter__(self):
        return self._read_files()


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
def main(args = None):
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
    set_seed(100)
    #set_seed(training_args.seed)

    config = oci.config.from_file()
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    fs = ocifs.OCIFileSystem(config=config)

    files = fs.ls(f'allenai_c4_en_raw@{namespace}')

    tokenizer = AutoTokenizer.from_pretrained("/nfs/cluster/models/meta-llama/Llama-3.1-70B/")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = C4OciIterableDataset(
        file_list=files,
        fs=fs,
        tokenizer=tokenizer,
        col_to_use="text",
        max_len=2048,
        truncation=True,
    )

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
    main()