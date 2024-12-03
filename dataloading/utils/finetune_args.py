from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DDPArguments:
    """
    Arguments required for distributed data parallel runs.
    """
    world_size: int = field(
        metadata={
            "help": "Total number of processes to launch across all nodes."
        }
    )
    local_world_size: int = field(
        metadata={
            "help": "Total number of processes to launch for each node."
        }
    )
    rank: int = field(
        metadata={
            "help": "Rank of current process, which is the range between 0 to WORLD_SIZE - 1."
        }
    )
    master_ip_addr: str = field(
        metadata={
            "help": "IP Address for the rank-zero process."
        }
    )
    master_port: int = field(
        metadata={
            "help": "The port for the rank-zero process."
        }
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        metadata={
            "help": "Path to pretrained model"
        }
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )


@dataclass
class DataTrainingArguments:
    bucket_name: str = field(
        metadata={
            "help": "Bucket containing dataset."
        }
    )
    local_cache_path: str = field(
        metadata={
            "help": "Path to local dataset cache to use while streaming."
        }
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=2048)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    oci_config_path: Optional[str] = field(
        default="~/.oci/config",
        metadata={
            "help": "Path to oci config, if not using default."
        }
    )
    oci_profile: Optional[str] = field(
        default="DEFAULT",
        metadata={
            "help": "OCI profile to use, if not using DEFAULT."
        }
    )
    local_cache_max_size_gbs: Optional[str] = field(
        default="25gb",
        metadata={
            "help": "Max size of dataset cache while streaming in gbs."
        }
    )
