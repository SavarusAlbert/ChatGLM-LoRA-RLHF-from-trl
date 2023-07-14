import torch
import numpy as np
import math
from datasets import load_dataset
from reward_model import reward_model
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, AutoModel, Seq2SeqTrainingArguments, set_seed, DataCollatorForSeq2Seq, HfArgumentParser
from transformers.optimization import get_scheduler
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from ..sft.arguments import ModelArguments, DataTrainingArguments
import torch.nn.utils.rnn as rnn_utils
from peft import LoraConfig, TaskType
from ppotrainer import PPOTrainerToRLHF, AutoModelForSeq2SeqLMWithValueHeadToRLHF

import tensorboard


def main():
    device = "cuda:0"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # set LoRA config
    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=model_args.trainable.split(','),
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )

    model = AutoModelForSeq2SeqLMWithValueHeadToRLHF.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, peft_config=peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)


    # load sft parameters
    lora_params = torch.load(data_args.sft_params)

    new_lora_params = dict()
    for key, value in lora_params.items():
        new_key = key.replace("base_model.model.transformer", "pretrained_model.base_model.model.transformer")
        new_lora_params[new_key] = value

    model.load_state_dict(new_lora_params, strict=False)
    # model.load_state_dict(lora_params, strict=False)

    # load value_head parameters
    # value_head_params = torch.load(data_args.value_head_params)
    # model.load_state_dict(value_head_params, strict=False)

    model = model.half().cuda()

    # freeze the LoRA layers
    for n, p in model.named_parameters():
        if n != "v_head.summary.weight" and "v_head.summary.bias":
            p.requires_grad = False

    # unfreeze part of layers
    unfreeze_layer_num = [22, 23, 24, 25, 26, 27]
    lora_layer = []
    for i in unfreeze_layer_num:
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.attention.query_key_value.lora_A.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.attention.query_key_value.lora_B.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.attention.dense.lora_A.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.attention.dense.lora_B.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.mlp.dense_h_to_4h.lora_A.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.mlp.dense_h_to_4h.lora_B.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.mlp.dense_4h_to_h.lora_A.default.weight")
        lora_layer.append(f"pretrained_model.base_model.model.transformer.layers.{i}.mlp.dense_4h_to_h.lora_B.default.weight")

    for n, p in model.named_parameters():
        if n in lora_layer:
            p.requires_grad = True


    # value_head initialize
    torch.nn.init.kaiming_uniform_(model.v_head.summary.weight, mode='fan_in', nonlinearity='relu')
    torch.nn.init.constant_(model.v_head.summary.bias, 0.0)


    delattr(model, "is_encoder_decoder")

    # create reference model
    ref_model = create_reference_model(model)


    ppo_config = PPOConfig(
        model_name=training_args.model_name,
        learning_rate=training_args.learning_rate,
        mini_batch_size=training_args.mini_batch_size,
        batch_size=training_args.batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        early_stopping=training_args.early_stopping,
        target_kl=training_args.target_kl,
        ppo_epochs=training_args.ppo_epochs,
        remove_unused_columns=False,
        log_with="tensorboard",
        accelerator_kwargs={"logging_dir": "./log"}
    )


    data_files = {}
    data_files["train"] = data_args.rlhf_file
    data_files["validation"] = data_args.validation_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )


    def ppo_preprocess_function_train(preds):
        inputs, labels, task_dataset = preds["input"], preds["target"], preds['task_dataset']
        input_ids = tokenizer.encode(inputs,
                            max_length=data_args.max_source_length,
                            truncation=True,
                            padding="max_length")
        label_ids = tokenizer.encode(labels, max_length=data_args.max_source_length, truncation=True, padding="max_length")
        task_dataset_ids = tokenizer.encode(task_dataset, max_length=data_args.max_source_length, truncation=True, padding="max_length")

        preds["input_ids"] = input_ids
        preds["label"] = label_ids
        preds['task_dataset_ids'] = task_dataset_ids
        return preds


    rlhf_dataset = raw_datasets["train"]
    column_names = raw_datasets["train"].column_names
    # column_names.remove('task_dataset')

    with training_args.main_process_first():
        rlhf_dataset = rlhf_dataset.map(
            ppo_preprocess_function_train,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )


    # label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=None,
            # padding=False
        )


    optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=training_args.learning_rate,
            # clip_threshold=10,
        )
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate)

    total_train_batch_size = training_args.mini_batch_size * training_args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=(training_args.ppo_epochs * math.ceil(len(rlhf_dataset) / total_train_batch_size))
        )

    ppo_trainer = PPOTrainerToRLHF(
        # training_args=training_args,
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=rlhf_dataset,
        data_collator=data_collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    generation_kwargs = {
        "min_length": -1,
        "max_length": 512,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "early_stopping": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # size = (batch_size, sequence_length), dtype = LongTensor
        inputs_ids = batch["input_ids"]
        inputs_tensors = inputs_ids.clone().detach().to(device)
        # inputs_tensors = torch.tensor(inputs_ids, dtype=torch.long).to(device)
        inputs_tensor_list = torch.split(inputs_tensors, 1, dim=0)
        inputs_tensor_list = [tensor.squeeze() for tensor in inputs_tensor_list]

        response_tensors = ppo_trainer.generate(
            inputs_tensor_list,
            return_prompt=False,
            # length_sampler=output_length_sampler,
            temperature=0.9,
            **generation_kwargs
        )

        response = []
        for res in response_tensors:
            try:
                response.append(tokenizer.batch_decode([res], skip_special_tokens=True)[0])
            except Exception as e:
                mask = res.lt(tokenizer.vocab_size)
                filtered_res_tensor = torch.masked_select(res, mask)
                response.append(tokenizer.batch_decode([filtered_res_tensor], skip_special_tokens=True)[0])

        batch["response"] = response
        batch["original"] = tokenizer.batch_decode(batch["label"])
        batch["task_dataset"] = tokenizer.batch_decode(batch["task_dataset_ids"])

        
        reward = reward_model(batch["response"], batch["original"], batch["task_dataset"])

        max_length=inputs_tensor_list[0].size()[0]

        for i in range(len(response_tensors)):
            tensor_length = len(response_tensors[i])
            if tensor_length > max_length:
                response_tensors[i] = response_tensors[i][:max_length]
            else:
                pads = torch.tensor([tokenizer.pad_token_id] * (max_length - tensor_length)).to(response_tensors[i].device)
                response_tensors[i] = torch.cat([response_tensors[i], pads], dim=0)


        padded_responses = rnn_utils.pad_sequence(response_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_responses_list = list(torch.unbind(padded_responses, dim=0))

        for i in range(len(inputs_tensor_list)):
            inputs_tensor_list[i] = inputs_tensor_list[i].long()
            padded_responses_list[i] = padded_responses_list[i].long()

        
        stats = ppo_trainer.step(queries=inputs_tensor_list, responses=padded_responses_list, scores=reward)
        ppo_trainer.log_stats(stats, batch, reward)
        
        if training_args.save_freq and epoch and epoch % training_args.save_freq == 0:
            trainable_params = {
                k: v for k, v in model.named_parameters() if "v_head" in k or "lora" in k
            }
            # save ppo parameters
            torch.save(trainable_params, f"trainable_params_step_{epoch}.pth")


if __name__ == "__main__":
    main()