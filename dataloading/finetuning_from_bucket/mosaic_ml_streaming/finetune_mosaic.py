import logging
import os
import sys
import time
from typing import Optional, Any

import torch
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.nn.utils.rnn import pad_sequence

from peft import get_peft_model

## MosaicML imports
from streaming import StreamingDataLoader, StreamingDataset

from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer
from transformers import set_seed

import oci
from oci.object_storage import ObjectStorageClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model, ProfilerCallback

#-------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('streaming')
logger.setLevel(logging.INFO)


#-------------------------------------------------------------------------------
class MosaicMLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #---------------------------------------------------------------------------
    def get_train_dataloader(self) -> StreamingDataLoader:
        """ Override the dataloader in transformers.Trainer() """
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        #return self.accelerator.prepare(StreamingDataLoader(train_dataset, **dataloader_params))
        return StreamingDataLoader(train_dataset, **dataloader_params)


    #---------------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """ Overriding loss function because of shape of llama output"""
        # Forward pass
        outputs = model(**inputs)
        
        # Extract per-token loss
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Reduce loss to scalar (average over batch and sequence length)
        if loss.dim() > 1:
            loss = loss.mean()
        
        return (loss, outputs) if return_outputs else loss

#-------------------------------------------------------------------------------
class StreamingC4Dataset(StreamingDataset):
    """ Init c4 dataset correctly. """
    def __init__(self,
                 *,
                 remote: Optional[str] = None,
                 local: Optional[str] = None,
                 split: Optional[str] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 keep_zip: bool = False,
                 epoch_size: Optional[int] = None,
                 predownload: Optional[int] = None,
                 cache_limit: Optional[int] = None,
                 partition_algo: str = 'orig',
                 num_canonical_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 shuffle_algo: str = 'py1s',
                 shuffle_seed: int = 9176,
                 shuffle_block_size: int = 1 << 18,
                 tokenizer: AutoTokenizer,
                 max_seq_len: int) -> None:

        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         keep_zip=keep_zip,
                         epoch_size=epoch_size,
                         predownload=predownload,
                         cache_limit=cache_limit,
                         partition_algo=partition_algo,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         shuffle_algo=shuffle_algo,
                         shuffle_seed=shuffle_seed,
                         shuffle_block_size=shuffle_block_size)

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    #---------------------------------------------------------------------------
    def _tokenize(self, sample: dict[str, Any]):
        """ Apply the tokenizer """
        return self.tokenizer(sample['text'],
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_seq_len,
                              return_tensors="pt")
    
    def get_item(self, idx: int) -> Any:
        """ Get sample by global index, blocking to load its shard if missing."""
        text_sample = super().get_item(idx)
        sample_len = len(text_sample["text"])
        url_len = len(text_sample["url"])
        timestamp_len = len(text_sample["timestamp"])
        tokenized_sample = self._tokenize(text_sample)
        input_ids = tokenized_sample["input_ids"].squeeze(0)
        attention_mask = tokenized_sample["attention_mask"].squeeze(0)
        # Might need a "labels" here.
        rank = get_rank()
        logger.info(f"Sample: {idx}, rank: {rank}, sample_length: {sample_len}, url_length: {url_len}, timestamp_len: {timestamp_len}, type: {input_ids.dtype}, shape: {input_ids.shape[0]}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    

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
def main(model_args, data_args, training_args, ddp_args):
    if not is_initialized():
        print("Distributed process not initialized.")
        return

    if model_args.profile:
        logger.info(f"{model_args=}")
        logger.info(f"{data_args=}")
        logger.info(f"{ddp_args=}")
        logger.info(f"{training_args=}")
    assert isinstance(training_args.gradient_accumulation_steps, int), "Invalid gradient_accumulation_steps"
    rank = get_rank()
    world_size = get_world_size()
    logger.info(f"Rank {rank}/{world_size} initialized successfully.")
    logger.info(f"{os.environ['WORLD_SIZE']=}")
    logger.info(f"{os.environ['LOCAL_WORLD_SIZE']=}")
    logger.info(f"{os.environ['RANK']=}")
    logger.info(f"{os.environ['MASTER_ADDR']=}")
    logger.info(f"{os.environ['MASTER_PORT']=}")
    logger.info(f"{os.environ['LOCAL_RANK']=}")
    set_seed(training_args.seed)
    config = oci.config.from_file(file_location=data_args.oci_config_path,
                                  profile_name=data_args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data

    remote_bucket = f'oci://{data_args.bucket_name}@{namespace}/'
    logger.info(f"Initializing StreamingDataset with remote={remote_bucket}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = StreamingC4Dataset(local=data_args.local_cache_path,
                                 remote=remote_bucket,
                                 download_retry=3,
                                 download_timeout=120,
                                 predownload=(training_args.per_device_train_batch_size * 8),
                                 batch_size=training_args.per_device_train_batch_size,
                                 shuffle=True,
                                 cache_limit=data_args.local_cache_max_size_gbs,
                                 num_canonical_nodes=(ddp_args.world_size // ddp_args.local_world_size),
                                 shuffle_seed=training_args.seed,
                                 shuffle_algo='py1e',
                                 tokenizer=tokenizer,
                                 max_seq_len=data_args.max_seq_length)

    model, _, peft_config = create_and_prepare_model(model_args)

    model = get_peft_model(model, peft_config)
    model.config.use_cache = not training_args.gradient_checkpointing
    model.config.return_dict = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    trainer = MosaicMLTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=collate_fn
    )
    start = time.perf_counter()
    if model_args.profile:
        logger.info("Running profiler.")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                                schedule=torch.profiler.schedule(skip_first=3,
                                                                                wait=1,
                                                                                warmup=1,
                                                                                active=2,
                                                                                repeat=1),
                                                on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-training-trainer'),
                                                profile_memory=True) as prof:
            trainer.add_callback(ProfilerCallback(prof=prof))
            trainer.train()
    else:
        trainer.train()
        trainer.save_model(training_args.output_dir)
    logger.info(f'Training time: {(time.perf_counter() - start):.1f}s')


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DDPArguments)
    )
    model_args, data_args, training_args, ddp_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.local_cache_path, exist_ok=True)
    logging.info(f"Created {data_args.local_cache_path}")
    main(model_args, data_args, training_args, ddp_args)