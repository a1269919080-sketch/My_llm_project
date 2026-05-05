import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # 使用国内镜像
from dataclasses import dataclass, field
from typing import Optional,Sequence,Dict
import transformers
from peft import LoraConfig, get_peft_model, TaskType
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import torch
#--------------------------------------------------------------------------------
#初始化
@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="",
        metadata={"help":"model name"}
    ) 
    tokenizer_name: Optional[str] = field(
        default="",
        metadata={"help":"tokenizer name"}
    )
@dataclass
class DatasetArguments:
    dataset_name: Optional[str] = field(
        default = "",
        metadata = {"help":"this is datasetname or dataset file"}
    )
    subset: Optional[str] = field(
        default = "",
        metadata = {"help":"this is subset name,if subset is none,keep the name none"}
    )
    load_from_json: Optional[bool] = field(
        default = False,
        metadata={"help": "whether to load dataset from json file"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})
parser = transformers.HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#--------------------------------------------------------------------------------
#模型
tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast = True
)
tokenizer.pad_token = tokenizer.eos_token
model  = AutoModelForCausalLM.from_pretrained(model_args.model_name, dtype=torch.float16, device_map="auto")

# LoRA配置
lora_config = LoraConfig(
    r=8,                         # 秩（越大越强，但更耗显存）
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # ⭐关键：Qwen必须写这个
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
#--------------------------------------------------------------------------------
#数据处理函数
IGNORE_INDEX=-100
def preprocess_function(examples):
    all_input_ids = []
    all_labels = []

    for messages in examples["messages"]:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        source_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        tokenized_full   = tokenizer(full_text,   return_attention_mask=False)
        tokenized_source = tokenizer(source_text, return_attention_mask=False)

        source_len = len(tokenized_source["input_ids"])

        input_ids = tokenized_full["input_ids"]                              # ✅ 直接用 list
        labels    = [IGNORE_INDEX] * source_len + input_ids[source_len:]     # ✅ 直接用 list

        assert len(input_ids) == len(labels)
        all_input_ids.append(input_ids)
        all_labels.append(labels)
    result = {'input_ids': all_input_ids, 'labels': all_labels}
    return result

#--------------------------------------------------------------------------------
#数据
#如果从json加载数据，则第一个，如果不是json加载数据，则走第二个
if data_args.load_from_json == True:
    train_dataset = load_dataset('json', data_files=data_args.dataset_name, split="train")
else:
    train_dataset = load_dataset(data_args.dataset_name, data_dir=data_args.subset, split="train_sft")

original_columns = train_dataset.column_names
dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size = 3000,
    remove_columns = original_columns,
    num_proc=16
)
#--------------------------------------------------------------------------------
#训练
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
)
model.config.use_cache = False

trainer.train()
trainer.save_model(training_args.output_dir)
trainer.save_state()