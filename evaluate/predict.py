import json
import torch
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments, HfArgumentParser

from ..sft.arguments import ModelArguments, DataTrainingArguments
from ..sft.main import load_pretrain_model, set_lora_config


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


config, tokenizer, model = load_pretrain_model(model_args)
model = set_lora_config(model, model_args)


# load ppo parameters
lora_params = torch.load(data_args.ppo_params)

new_lora_params = dict()
for key, value in lora_params.items():
    new_key = key.replace("pretrained_model.base_model.model.transformer", "base_model.model.transformer")
    new_lora_params[new_key] = value

model.load_state_dict(new_lora_params, strict=False)


model = model.half().cuda().eval()


# test dataset
testA_dataset = []
for line in open(data_args.test_file, "r", encoding="utf-8"):
    testA_dataset.append(json.loads(line))


# predict
submit_prediction = []
for idx in tqdm(range(len(testA_dataset))):
    data = testA_dataset[idx]
    input_ids = data["input"]

    response, history = model.chat(tokenizer, input_ids, history=[]) #, temperature=0.01)
        
    data["target"] = response
    submit_prediction.append(data)


# save result
with open("test_predictions.json", "w", encoding="utf-8") as file:
    for d in submit_prediction:
        json.dump(d, file, ensure_ascii=False)
        file.write('\n')