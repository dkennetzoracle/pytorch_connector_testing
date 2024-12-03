from dataclasses import dataclass, field
import os
import sys
from typing import Optional

## MosaicML imports
from streaming import StreamingDataLoader

from transformers import HfArgumentParser, TrainingArguments
from transformers import set_seed
from trl import SFTTrainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model, create_mosaic_ml_streaming_dataset

class MosaicMLTrainer(SFTTrainer):
    def get_train_dataloader(self) -> StreamingDataLoader:
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True
        )

def setup(ddp_args):
    os.environ['WORLD_SIZE'] = str(ddp_args.world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(ddp_args.local_world_size)
    os.environ['RANK'] = str(ddp_args.rank)
    os.environ['MASTER_ADDR'] = ddp_args.master_ip_addr
    os.environ['MASTER_PORT'] = str(ddp_args.master_port)

def main(model_args, data_args, training_args, ddp_args):
    setup(ddp_args)
    set_seed(training_args.seed)
    model, tokenizer, peft_config = create_and_prepare_model(model_args)
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }
    
    train_dataset = create_mosaic_ml_streaming_dataset(tokenizer, data_args, training_args, ddp_args)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,

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
    main(model_args, data_args, training_args, ddp_args)