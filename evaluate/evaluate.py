import torch
import numpy as np
import json
from ..rlhf.reward_model import reward_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Seq2SeqTrainingArguments, HfArgumentParser, set_seed
from transformers.optimization import get_scheduler
from ..sft.arguments import ModelArguments, DataTrainingArguments
from ..sft.main import set_lora_config



device = "cuda:0"

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


set_seed(training_args.seed)
torch.manual_seed(training_args.seed)
np.random.seed(training_args.seed)


model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
model = set_lora_config(model, model_args)

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)


# checkpoint = f"checkpoint-****/pytorch_model.bin"
# model.load_state_dict(torch.load(checkpoint))

# load ppo parameters
lora_params = torch.load(data_args.ppo_params)

new_lora_params = dict()
for key, value in lora_params.items():
    new_key = key.replace("pretrained_model.base_model.model.transformer", "base_model.model.transformer")
    new_lora_params[new_key] = value

model.load_state_dict(new_lora_params, strict=False)


model = model.half().cuda()

eval_dataset = []
for line in open(data_args.validation_file, "r", encoding="utf-8"):
    eval_dataset.append(json.loads(line))


submit_prediction = []
for idx in tqdm(range(len(eval_dataset))):
    data = eval_dataset[idx]
    input_ids = data["input"]

    response, history = model.chat(tokenizer, input_ids, history=[])
        
    data["response"] = response
    submit_prediction.append(data)


reward = reward_model([data["target"] for data in submit_prediction], [data["target"] for data in submit_prediction], [data["task_dataset"] for data in submit_prediction])

print([r for r in reward if r != 1])
print(len([x for x in reward if x != 1]))

