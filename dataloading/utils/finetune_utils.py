from typing import Tuple

from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def create_and_prepare_model(model_args) -> Tuple[AutoModelForCausalLM, AutoTokenizer, LoraConfig]:
    """ Setup a model for fine-tuning. """
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
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