import logging
import os
import sys
import tempfile
import shutil

## MosaicML imports
from streaming import StreamingDataLoader, StreamingDataset

from transformers import HfArgumentParser, TrainingArguments
from transformers import set_seed
from trl import SFTTrainer

import oci
from oci.object_storage import ObjectStorageClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MosaicMLTrainer(SFTTrainer):
    def get_train_dataloader(self) -> StreamingDataLoader:
        return self.accelerator.prepare(StreamingDataLoader(
            self.train_dataset,
            batch_size=self.dataset_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True
        ))

def setup(ddp_args):
    os.environ['WORLD_SIZE'] = str(ddp_args.world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(ddp_args.local_world_size)
    os.environ['RANK'] = str(ddp_args.rank)
    os.environ['MASTER_ADDR'] = ddp_args.master_ip_addr
    os.environ['MASTER_PORT'] = str(ddp_args.master_port)
    print(f"{os.environ['WORLD_SIZE']=}")
    print(f"{os.environ['LOCAL_WORLD_SIZE']=}")
    print(f"{os.environ['RANK']=}")
    print(f"{os.environ['MASTER_ADDR']=}")
    print(f"{os.environ['MASTER_PORT']=}")

def main(model_args, data_args, training_args, ddp_args):
    setup(ddp_args)
    set_seed(training_args.seed)
    model, tokenizer, peft_config = create_and_prepare_model(model_args)
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }
    
    config = oci.config.from_file(file_location=data_args.oci_config_path,
                                  profile_name=data_args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    obj = object_storage_client.get_object(namespace_name=namespace,
                                           bucket_name=data_args.bucket_name,
                                           object_name="index.json")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = tmp_file.name
        for chunk in obj.data.raw.stream(1024 * 1024, decode_content=False):
            tmp_file.write(chunk)
        index_file = os.path.join(data_args.local_cache_path, "index.json")
        if not os.path.exists(index_file):
            shutil.move(temp_path, index_file)
        print(f"Wrote {index_file}")
    remote_bucket = f'oci://{data_args.bucket_name}@{namespace}/'
    logger.info(f"Initializing StreamingDataset with remote={remote_bucket}")
    dataset = StreamingDataset(local=data_args.local_cache_path,
                               remote=remote_bucket,
                               download_retry=3,
                               download_timeout=120,
                               batch_size=data_args.batch_size,
                               shuffle=True,
                               cache_limit=data_args.local_cache_max_size_gbs,
                               num_canonical_nodes=(ddp_args.world_size // ddp_args.local_world_size),
                               shuffle_seed=training_args.seed,
                               shuffle_algo='py1e'
                               )
    logger.info("StreamingDataset initialized successfully")
    train_dataset = dataset.map(
        lambda samples: tokenizer(samples['text'],
                                  max_length=data_args.max_seq_length,
                                  truncation=True,
                                  ), batched=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_batch_size=data_args.batch_size

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
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DDPArguments)
    )
    model_args, data_args, training_args, ddp_args = parser.parse_args_into_dataclasses()
    os.makedirs(data_args.local_cache_path, exist_ok=True)
    logging.info(f"Created {data_args.local_cache_path}")
    main(model_args, data_args, training_args, ddp_args)