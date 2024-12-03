from dataclasses import dataclass, field
import os
import sys
from typing import Optional

## MosaicML imports
from streaming import StreamingDataLoader

from transformers import HfArgumentParser, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import set_seed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.finetune_args import ModelArguments, DataTrainingArguments, DDPArguments
from utils.finetune_utils import create_and_prepare_model, create_mosaic_ml_streaming_dataset

class MosaicMLTrainer(Trainer):
    def get_train_dataloader(self) -> StreamingDataLoader:
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers
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
    model, tokenizer = create_and_prepare_model(model_args)
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }
    
    train_dataset = create_mosaic_ml_streaming_dataset(tokenizer, data_args, training_args)

    trainer = MosaicMLTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        )
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, DDPArguments)
    )
    model_args, data_args, training_args, ddp_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args, ddp_args)