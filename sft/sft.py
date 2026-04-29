import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 使用国内镜像
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import (
    AutoModelForCausalLM,
    
    AutoTokenizer,
    
    
)
import torch
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="",
        metadata={"help":"the model name or path"}
    )
    tokenizer_name: Optional[str] = field(
        default="",
        metadata={"help":"the tokenizer for your model, if left enpty will use the default for your model"}
    )
@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="", metadata={"help": "the subset to use"})
    load_from_json: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load dataset from json file"},
    )
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})

parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#一般模型都有配套的tokenizer，这里防止想用另外的tokenizer
tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name
#初始化一个tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast=True,
    #trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
num_proc = 16
model = AutoModelForCausalLM.from_pretrained(model_args.model_name)

#模型并行
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

def preprocess_function(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    








