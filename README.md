# ChatGLM-LoRA-RLHF

## Competition address
- [CCKS2023-PromptCBLUE中文医疗大模型评测基准—开源赛道](https://tianchi.aliyun.com/competition/entrance/532084/introduction?spm=a2c22.12281949.0.0.4c885d9b4WyXT4)

## Dataset
- [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE.git)

## Fine-Tuning Methods
- LoRA
  - Fine-tuning the low-rank adapters of the ChatGLM-6B model.
- Reward Model
  - Building a reward model according to dataset.
- RLHF
  - Fine-tuning the low-rank adapters of LM with reinforcement learning.

## Getting Started

### Dependence Installation
```bash
git clone https://github.com/SavarusAlbert/ChatGLM-LoRA-RLHF-from-trl.git
```
- install packages
```bash
pip install -r requirements.txt
```

### Fine-tuning with a Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/chatglm-6b-lora-$PRE_SEQ_LEN \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "epoch" \
    --num_train_epochs 10 \
    --logging_steps 10 \
    --learning_rate 1e-4 \
    --warmup_steps 20 \
    --training_args.weight_decay 0.0001
```

### Training with RLHF

```bash
CUDA_VISIBLE_DEVICES=0 python rlhf.py \
    --do_train \
    --model_name None \
    --output_dir path_to_ppo_checkpoint \
    --mini_batch_size 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --early_stopping None \
    --target_kl 0.1 \
    --warmup_steps 5 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1.45e-5 \
    --ppo_epochs 1
```

## TODO
- [ ] Employing [LangChain](https://github.com/hwchase17/langchain) to easily build applications that are capable of leveraging external knowledge upon fine-tuned ChatGLM models.
- [ ] Using more RLHF packages to implement reinforcement learning.
  - [x] [TRL](https://github.com/lvwerra/trl.git)
  - [ ] [TRLx](https://github.com/CarperAI/trlx.git)
  - [ ] [Safe-rlhf](https://github.com/PKU-Alignment/safe-rlhf.git)
  - [ ] [RL4LMs](https://github.com/allenai/RL4LMs.git)

## Citation

If this work is helpful, please cite as:

```bibtex
@Misc{ChatGLM-LoRA-RLHF,
  title = {ChatGLM LoRA RLHF},
  author = {SavarusAlbert},
  howpublished = {\url{https://github.com/SavarusAlbert/ChatGLM-LoRA-RLHF-from-trl}},
  year = {2023}
}
```

## Acknowledgement

This repo benefits from [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and [TRL](https://github.com/lvwerra/trl.git). Thanks for their wonderful works.