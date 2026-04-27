import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from dataclasses import dataclass, field
from typing import Optional


print("正在加载基础模型...")

@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="",
        metadata={"help": "the model name or path"}
    )
    tokenizer_name: Optional[str] = field(
        default="",
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model"
        }
    )
parser = transformers.HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()[0]
tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast=True,
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

chat_history = []
print("\n" + "="*50)
print("Gemma-2 模型已加载完成！你可以开始和她聊天了。")
print("输入 'quit', 'exit' 或 '退出' 结束对话，输入 'clear' 清空记忆。")
print("="*50 + "\n")

while True:
   user_input = input("\n你: ") 
   if user_input.strip().lower() in ['quit', 'exit', '退出']:
        print("对话结束。")
        break 
   if user_input.strip().lower() in ['clear', '清空']:
        chat_history = []
        print("（已清空历史记忆）")
        continue
   chat_history.append({"role": "user", "content": user_input})

   prompt = tokenizer.apply_chat_template(
        chat_history, 
        tokenize=False, 
        add_generation_prompt=True
   )
   inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
   print("Gemma: ", end="")
   with torch.no_grad(): # 科研推理时一定要加这句，节省显存
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,       # 控制回答的最大长度
            temperature=0.7,          # 控制回答的随机性（0.1-1.0之间）
            streamer=streamer,        # 启用打字机效果
            pad_token_id=tokenizer.eos_token_id
        )
   input_length = inputs['input_ids'].shape[1]
   generated_tokens = outputs[0][input_length:]
   model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
   chat_history.append({"role": "model", "content": model_response})
