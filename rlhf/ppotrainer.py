import torch

from trl.trainer.ppo_trainer import logprobs_from_logits, PreTrainedModelWrapper, PPODecorators, PPOTrainer

from trl import AutoModelForSeq2SeqLMWithValueHead, AutoModelForCausalLMWithValueHead


class AutoModelForSeq2SeqLMWithValueHeadToRLHF(AutoModelForSeq2SeqLMWithValueHead):
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=True,  # We force the model to output hidden states
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        value = self.v_head(last_hidden_state).squeeze(-1)

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)





class PPOTrainerToRLHF(PPOTrainer):
    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_data = self.data_collator(
            # [{"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.bool)} for ids in input_ids]
            [{"input_ids": ids} for ids in input_ids]
        ).to(self.current_device)

        input_data.pop("labels", None)

        return input_data
    
    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(input_ids)
            for i, ids in enumerate(input_ids):
                start_i = (ids == self.model.config.bos_token_id).nonzero()
                end_i = (ids == self.model.config.eos_token_id).nonzero()

                if start_i.size() != torch.tensor(1):
                    start_i = start_i[0]

                if len(end_i):
                    end_i = end_i[0][0] + 1
                else:
                    end_i = None
                masks[i][start_i:end_i] = 1

            all_logits.append(logits)
            all_values.append(values.rot90())
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1],
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )
