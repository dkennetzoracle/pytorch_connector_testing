import logging
import os
import sys
import tempfile
import shutil

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

import torch
from peft import get_peft_model

## MosaicML imports
from streaming import StreamingDataLoader, StreamingDataset
from streaming.text import StreamingC4

from transformers import HfArgumentParser, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import set_seed
from trl import SFTTrainer

import oci
from oci.object_storage import ObjectStorageClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model

#--------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('streaming')
logger.setLevel(logging.INFO)

class MosaicMLTrainer(Trainer):
    def __init__(self, streaming_batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_batch_size = streaming_batch_size
    def get_train_dataloader(self) -> StreamingDataLoader:
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.streaming_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.data_collator
        )    

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([torch.tensor(mask) for mask in attention_mask], batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def main(model_args, data_args, training_args, ddp_args):
    print(f"{os.environ['WORLD_SIZE']=}")
    print(f"{os.environ['LOCAL_WORLD_SIZE']=}")
    print(f"{os.environ['RANK']=}")
    print(f"{os.environ['MASTER_ADDR']=}")
    print(f"{os.environ['MASTER_PORT']=}")
    print(f"{os.environ['LOCAL_RANK']=}")
    set_seed(training_args.seed)
    config = oci.config.from_file(file_location=data_args.oci_config_path,
                                  profile_name=data_args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data

    remote_bucket = f'oci://{data_args.bucket_name}@{namespace}/'
    logger.info(f"Initializing StreamingDataset with remote={remote_bucket}")
    dataset = StreamingC4(local=data_args.local_cache_path,
                          remote=remote_bucket,
                          download_retry=3,
                          download_timeout=120,
                          #predownload=(data_args.batch_size * 8),
                          batch_size=data_args.batch_size,
                          shuffle=True,
                          cache_limit=data_args.local_cache_max_size_gbs,
                          num_canonical_nodes=(ddp_args.world_size // ddp_args.local_world_size),
                          shuffle_seed=training_args.seed,
                          shuffle_algo='py1e',
                          tokenizer_name=model_args.model_path,
                          group_method='truncate',
                          max_seq_len=data_args.max_seq_length)

    ## Debuggery
    """ loader = StreamingDataLoader(
        dataset,
        batch_size=data_args.batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=collate_fn
    )
    num_samples = 0
    for data in tqdm(loader, smoothing=0, mininterval=1):
        print(f"{data=}")
        num_samples += 1
        if num_samples == 1:
            sys.exit()
    logger.info("StreamingDataset initialized successfully") """

    model, tokenizer, peft_config = create_and_prepare_model(model_args)
    tokenizer.pad_token = tokenizer.eos_token
    model = get_peft_model(model, peft_config)
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    trainer = MosaicMLTrainer(
        streaming_batch_size=data_args.batch_size,
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=collate_fn
    )

    """ trainer = MosaicMLTrainer(
        streaming_batch_size=data_args.batch_size,
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        )
    ) """
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(training_args.output_dir)
    """ trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_batch_size=data_args.batch_size,
        dataset_num_proc=(os.cpu_count() // 4)
    )
    trainer.accelerator.print(f"{trainer.model}")
    if model_args.use_peft_lora:
        trainer.model.print_trainable_parameters()
    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir) """


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DDPArguments)
    )
    model_args, data_args, training_args, ddp_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.local_cache_path, exist_ok=True)
    logging.info(f"Created {data_args.local_cache_path}")
    main(model_args, data_args, training_args, ddp_args)