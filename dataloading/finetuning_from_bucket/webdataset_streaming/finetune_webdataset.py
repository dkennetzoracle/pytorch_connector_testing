import logging
import os
import sys
import time
from typing import List

import oci
from oci.object_storage import ObjectStorageClient

from peft import get_peft_model

import torch
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.nn.utils.rnn import pad_sequence

from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, Trainer
from transformers import set_seed

import webdataset as wds
from webdataset import WebDataset, WebLoader

# Local imports
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
class WebDatasetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    #---------------------------------------------------------------------------
    def get_train_dataloader(self) -> WebLoader:
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

        trainloader = WebLoader(train_dataset,
                                batch_size=None,
                                num_workers=self.args.dataloader_num_workers)
        trainloader = trainloader.unbatched().shuffle(self.args.seed).batched(self._train_batch_size)
        return WebLoader(train_dataset, **dataloader_params)
    
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
def list_all_objects_in_bucket(object_storage_client: ObjectStorageClient, namespace: str, bucket: str, par: str):
    objects = []
    resp = object_storage_client.list_objects(namespace_name=namespace, bucket_name=bucket)
    for object in resp.data.objects:
        objects.append(f'{par}{object.name}')
    
    while resp.data.next_start_with:
        resp = resp = object_storage_client.list_objects(namespace, bucket, start=resp.data.next_start_with)
        for object in resp.data.objects:
            objects.append(f'{par}{object.name}')

    #return [f"pipe: curl -L -s {object}" for object in objects if object.endswith(".tar")]
    return [object for object in objects if object.endswith(".tar")]

#-------------------------------------------------------------------------------
class TokenizeDataset:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, sample):
        # Assuming the 'text' key contains the text to tokenize
        tokenized_sample = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        sample_len = len(sample["text"])
        rank = get_rank()
        input_ids = tokenized_sample["input_ids"].squeeze(0)
        attention_mask = tokenized_sample["attention_mask"].squeeze(0)
        logger.info(f"Sample: 0, rank: {rank}, sample_length: {sample_len}, url_length: 30, timestamp_len: 19, type: {input_ids.dtype}, shape: {input_ids.shape[0]}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

#-------------------------------------------------------------------------------
def decode_text(sample):
    if "text" in sample:
        sample["text"] = sample["text"].decode("utf-8")  # Decode bytes to string
    return sample

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
    """https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb"""
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

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # par = "https://objectstorage.us-ashburn-1.oraclecloud.com/p/fwGBLrEs8TGnudnr0C_T3ZkhjC6Zkc4TTDCAJFz4DOZ_nDU1yGDl338AzSENndBJ/n/iduyx1qnmway/b/wds_hub_allenaic4en/o/"
    # "wds_hub_allenaic4en"
    config = oci.config.from_file(file_location=data_args.oci_config_path,
                                  profile_name=data_args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    shard_urls = list_all_objects_in_bucket(object_storage_client=object_storage_client,
                                            namespace=namespace,
                                            bucket=data_args.bucket_name,
                                            par=data_args.pre_authenticated_request)


    dataset = (
        wds.WebDataset(shard_urls, resampled=True, cache_dir=data_args.local_cache_path, nodesplitter=wds.split_by_node)
        .shuffle(training_args.seed)
        .map(decode_text)
        .map(TokenizeDataset(tokenizer, max_seq_len=data_args.max_seq_length))
    )

    model, _, peft_config = create_and_prepare_model(model_args)

    model = get_peft_model(model, peft_config)
    model.config.use_cache = not training_args.gradient_checkpointing
    model.config.return_dict = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    trainer = WebDatasetTrainer(
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

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DDPArguments)
    )
    model_args, data_args, training_args, ddp_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.local_cache_path, exist_ok=True)
    logging.info(f"Created {data_args.local_cache_path}")
    main(model_args, data_args, training_args, ddp_args)
