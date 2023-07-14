from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    # 模型参数
    model_name_or_path: str = field(default=None)
    ptuning_checkpoint: str = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir:Optional[str] = field(default="./")

    # LoRa参数
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.)

    # 配置输入前缀长度，投影层
    pre_seq_len: Optional[int] = field(
        default=None
    )
    prefix_projection: bool = field(
        default=False
    )

    # 可训练的参数
    trainable: Optional[str] = field(default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default="dataset/new_train.json")
    validation_file: Optional[str] = field(default="dataset/dev.json")
    test_file: Optional[str] = field(default="****.json")
    rlhf_file: Optional[str] = field(default="dataset/rlhf.json")

    prompt_column: Optional[str] = field(default="input")
    response_column: Optional[str] = field(default="target")
    history_column: Optional[str] = field(default=None)

    preprocessing_num_workers: Optional[int] = field(default=None)

    max_source_length: Optional[int] = field(default=256)
    max_target_length: Optional[int] = field(default=256)

    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=100)

    ignore_pad_token_for_loss: bool = field(default=True)
    # 配置前缀
    source_prefix: Optional[str] = field(default="")

    sft_params: Optional[str] = field(default="sft/lora_params.pth")
    ppo_params: Optional[str] = field(default="params_step_****.pth")
    value_head_params: Optional[str] = field(default="")

    log_dir : Optional[str] = field(default="./log/sft")
