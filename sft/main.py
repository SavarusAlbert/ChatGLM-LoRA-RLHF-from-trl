import torch
import jieba
import numpy as np
from datasets import load_dataset
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

from arguments import ModelArguments, DataTrainingArguments
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


def load_pretrain_model(model_args, use_half_precise=True, use_cuda=True):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if use_half_precise:
        model = model.half()
    if use_cuda:
        model = model.cuda()

    return config, tokenizer, model


def set_lora_config(model, model_args):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.trainable.split(','),
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    return get_peft_model(model, peft_config)


def get_spicial_tokens(tokenizer):
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    mask = tokenizer.mask_token_id
    gmask = tokenizer.gmask_token_id

    return bos, eos, pad, mask, gmask


def preprocess_function_train(tokenizer, data, data_column, data_args, prefix=None):
    prompt_column = data_column["prompt_column"]
    response_column = data_column["response_column"]
    history_column = data_column["history_column"]

    max_seq_length = data_args.max_source_length + data_args.max_target_length

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }

    bos, eos, pad, mask, gmask = get_spicial_tokens(tokenizer)
    
    for i in range(len(data[prompt_column])):
        if data[prompt_column][i] and data[response_column][i]:
            query, answer = data[prompt_column][i], data[response_column][i]

            if history_column is None:
                prompt = query
            else:
                prompt = ""
                history = data[history_column][i]
                for idx, (old_query, response) in enumerate(history):
                    prompt += f"[Round {idx}]\n问：{old_query}\n答：{response}\n"
                prompt += f"[Round {len(history)}]\n问：{query}\n答："
            
            if prefix:
                prompt = prefix + prompt
            input_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            label_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(input_ids) > data_args.max_source_length - 1:
                if prefix:
                    if len(prefix) > data_args.max_source_length - 50:
                        input_ids = input_ids[:data_args.max_source_length - 1]
                    else:
                        input_ids = input_ids[:len(prefix)] + input_ids[-(data_args.max_source_length - 1-len(prefix)):]
                else:
                    input_ids = input_ids[:data_args.max_source_length - 1]
            
            if len(label_ids) > data_args.max_target_length - 2:
                label_ids = label_ids[:data_args.max_target_length - 2]
            
            inputs = input_ids + [gmask] + [bos] + label_ids + [eos]

            context_length = len(input_ids) + 1
            labels = [-100] * context_length + [bos] + label_ids + [eos]

            pad_len = max_seq_length - len(inputs)
            inputs = inputs + [pad] * pad_len
            labels = labels + [pad] * pad_len

            if data_args.ignore_pad_token_for_loss:
                labels = [(l if l != pad else -100) for l in labels]
            
            model_inputs["input_ids"].append(inputs)
            model_inputs["labels"].append(labels)
        
    return model_inputs


def preprocess_function_eval(tokenizer, data, data_column, data_args, prefix=None):
    prompt_column = data_column["prompt_column"]
    response_column = data_column["response_column"]
    history_column = data_column["history_column"]

    inputs, targets = [], []
    for i in range(len(data[prompt_column])):
        if not data[response_column][i]:
            targets.append("filled in !")
        else:
            targets.append(data[response_column][i])
        
        if data[prompt_column][i]:
            query = data[prompt_column][i]
            if history_column is None or len(data[history_column][i]) == 0:
                prompt = query
            else:
                prompt = ""
                history = data[history_column][i]
                for idx, (old_query, response) in enumerate(history):
                    prompt += f"[Round {idx}]\n问：{old_query}\n答：{response}\n"
                prompt += f"[Round {len(history)}]\n问：{query}\n答："
            inputs.append(prompt)

    if prefix:
        inputs = [prefix + inp for inp in inputs]            
            
    input_ids = tokenizer(inputs,
                          max_length=data_args.max_source_length,
                          truncation=True,
                          padding="max_length")
    label_ids = tokenizer(text_target=targets, max_length=data_args.max_source_length, truncation=True, padding="max_length")

    if data_args.ignore_pad_token_for_loss:
        label_ids["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in label_ids["input_ids"]
        ]

    input_ids["labels"] = label_ids["input_ids"]
    return input_ids


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    # training_args.seed = 100
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # load model
    config, tokenizer, model = load_pretrain_model(model_args)

    # set LoRA config
    model = set_lora_config(model, model_args)

    # print trainable parameters
    if training_args.do_train:
        model.print_trainable_parameters()

    if training_args.do_eval:
        # load sft parameters
        # model.load_state_dict(torch.load(data_args.sft_params))

        # load ppo parameters
        lora_params = torch.load(data_args.ppo_params)

        new_lora_params = dict()
        for key, value in lora_params.items():
            new_key = key.replace("pretrained_model.base_model.model.transformer", "base_model.model.transformer")
            new_lora_params[new_key] = value

        model.load_state_dict(new_lora_params, strict=False)

    # load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file        
    raw_datesets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )


    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # train/eval/test column
    if training_args.do_train:
        column_names = raw_datesets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datesets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datesets["test"].column_names
    else:
        return

    data_column = {}
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    data_column["prompt_column"] = prompt_column
    data_column["response_column"] = response_column
    data_column["history_column"] = history_column
    

    if training_args.do_train:
        if "train" not in raw_datesets:
            raise ValueError("require a train dataset")
        train_dataset = raw_datesets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_simples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first():
            train_dataset = train_dataset.map(
                lambda x: preprocess_function_train(tokenizer, x, data_column, data_args, prefix=prefix),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
            )
    
    if training_args.do_eval:
        if "validation" not in raw_datesets:
            raise ValueError("requere a validation dataset")
        eval_dataset = raw_datesets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first():
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_function_eval(tokenizer, x, data_column, data_args, prefix=prefix),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=False,
            )


    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        # padding=False
    )


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]
        preds= np.argmax(preds, axis=-1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        print("decoded_preds:", decoded_preds)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("decoded_labels:", decoded_labels)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": [],
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    summary_writer = SummaryWriter(log_dir=data_args.log_dir)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[TensorBoardCallback(summary_writer)],
    )


    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=data_args.max_source_length+1, temperature=0.1)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
    