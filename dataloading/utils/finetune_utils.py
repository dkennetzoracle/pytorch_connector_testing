from typing import Tuple

from streaming import StreamingDataset, StreamingDataLoader

from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import oci
from oci.object_storage import ObjectStorageClient

def create_mosaic_ml_streaming_dataset(tokenizer, data_args, trainer_args):
    config = oci.config.from_file(file_location=data_args.oci_config_path,
                                  profile_name=data_args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    remote_bucket = f'oci://{data_args.bucket_name}@{namespace}/'
    dataset = StreamingDataset(local=data_args.local_cache_path,
                               remote=remote_bucket,
                               download_retry=3,
                               batch_size=trainer_args.train_batch_size,
                               shuffle=True,
                               cache_limit=data_args.local_cache_max_size,)
    dataset = dataset.map(
        lambda samples: tokenizer(samples['text'],
                                  max_length=data_args.max_seq_length,
                                  truncation=True,
                                  ), batched=True
    )
    return dataset

def create_and_prepare_model(model_args) -> Tuple[AutoModelForCausalLM, AutoTokenizer, LoraConfig]:
    """ Setup a model for fine-tuning. """
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float32
    )
    # Replace with args versions.
    peft_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules.split(","),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, peft_config