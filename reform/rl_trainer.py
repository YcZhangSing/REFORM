# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from tqdm import tqdm
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized, Dict
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from PIL import Image
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from reform.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import AriaForConditionalGeneration
except ImportError:
    AriaForConditionalGeneration = None

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


import math
def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class REFORMNoSelfRewardGRPOTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        greater_is_better: bool = True,
        best_metric_name: str = "eval_accuracy",
        torch_dtype: str = "bfloat16",
        compute_metrics: Optional[Callable[[tuple[list, list]], Dict[str, float]]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        
        self.compute_metrics = compute_metrics
        self.best_metric_name = best_metric_name
        self.greater_is_better = greater_is_better
        self.best_metric_value = float('-inf') if greater_is_better else float('inf')
        self.best_model_path = os.path.join(args.output_dir, "best_model") if args else "best_model"
        os.makedirs(self.best_model_path, exist_ok=True)
        
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            
            model_init_kwargs.pop("use_cache")
            model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, **model_init_kwargs,)
            # if "Qwen2-VL" in model_id:
            #     model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            # elif "Qwen2.5-VL" in model_id:
            #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            # elif "Aria" in model_id:
            #     model_init_kwargs.pop("use_cache")
            #     model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            # else:
            #     model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        self.vision_modules_keywords = ["visual"]
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)
        # pad_token_id = processing_class.pad_token_id
        pad_token_id = 1
        
        # if processing_class is None:
        #     if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
        #         processing_class = AutoProcessor.from_pretrained(model_id)
        #         pad_token_id = processing_class.tokenizer.pad_token_id
        #         processing_class.pad_token_id = pad_token_id
        #         processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        #         if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
        #             processing_class.image_processor.max_pixels = max_pixels
        #             processing_class.image_processor.min_pixels = min_pixels
        #     else:
        #         processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        #         pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        self.epsilon = args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    
    
    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    

    def evaluate(self, eval_dataset=None, batch_size=4, metric_key_prefix="eval", ignore_keys=None,):
        def collate_fn(batch, processing_class, device):
            prompts = []
            images = []
            labels = []

            for item in batch:
                prompts.append(item.get("problem", "ERR"))
                labels.append(item["solution"])

                # 处理图像
                if "image" in item:
                    img = item["image"]
                elif "image_path" in item and item["image_path"] is not None:
                    img = Image.open(item["image_path"])
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                else:
                    img = None

                if img is not None:
                    img = img.resize((768, 768), Image.Resampling.LANCZOS)
                images.append(img)

            inputs = processing_class(
                text=prompts,
                images=images if any(images) else None,
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].bfloat16()

            return inputs, labels
        
        eval_dataset = eval_dataset or self.eval_dataset
        all_preds = []
        all_labels = []

        model = self.model
        model.eval()
        device = self.accelerator.device

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, self.processing_class, device),
            num_workers=0,
        )

        for batch_inputs, batch_labels in tqdm(eval_loader, desc="Evaluating", ncols=100):
            with torch.no_grad():
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    pred_ids = unwrapped_model.generate(
                        input_ids=batch_inputs.get("input_ids"),
                        pixel_values=batch_inputs.get("pixel_values"),
                        max_new_tokens=1024,
                        num_beams=3,
                        return_decoder2_outputs=False,
                        For_RL=False
                    )
                    preds = self.processing_class.batch_decode(pred_ids, skip_special_tokens=False)
                    preds = [re.sub(r"<pad>|<s>|</s>", "", pred) for pred in preds]
                    # print(f'preds is {preds};  all_labels is {all_labels}')
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels)

        metrics = {}

        metrics = self.compute_metrics((all_preds, all_labels))
        
        # 保存 metrics 到文件
        if self.accelerator.is_main_process:
            metrics_log_path = os.path.join(self.args.output_dir, "eval_metrics_log.txt")
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(metrics_log_path, "a", encoding="utf-8") as f:
                f.write(f"Step {self.state.global_step}: {metrics}\n")

        # 只在主进程保存最佳模型
        if self.accelerator.is_main_process and self.best_metric_name in metrics:
            current_value = metrics[self.best_metric_name]
            is_better = current_value > self.best_metric_value if self.greater_is_better else current_value < self.best_metric_value
            if is_better:
                self.best_metric_value = current_value
                print(f"[MyGRPOTrainer] Saving new best model with {self.best_metric_name}: {current_value}")
                self.save_model(self.best_model_path)

        return metrics

    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values):
        
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def _get_per_token_logps_florence(self, model, decoder_input_ids, input_ids=None, attention_mask=None, pixel_values=None):
        """
        计算每个 token 的 log probability（针对 Florence2 encoder-decoder）。
        
        参数:
            model: Florence2 模型
            decoder_input_ids: tensor (B, L), decoder 输入序列
            attention_mask: tensor (B, L), decoder attention mask，可选
            pixel_values: tensor (B, C, 768, 768), encoder 图像输入
            
        返回:
            per_token_logps: tensor (B, L), 每个 token 的 log probability
        """
        # 调用模型，注意 Florence2 encoder-decoder 需要 decoder_input_ids。
        # Some custom Florence2 checkpoints assume decoder2_input_ids is a tensor whenever `self.training=True`
        # and may crash with "'bool' object has no attribute 'int'" when it's None.
        # We temporarily switch to eval mode for this rollout/logprob forward, while keeping autograd enabled.
        was_training = model.training
        if was_training:
            model.eval()
        try:
            ## 目前只需要decoder_input_ids 即 主decoder对answer概率即可，对于辅助decoder的概率获取与优化暂时不考虑
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask,
                return_dict=True,
            )
        finally:
            if was_training:
                model.train()
        
        logits = outputs.logits  # (B, L, V)
        
        # 排除最后一个 logit，因为它预测的是下一个 token
        logits = logits[:, :-1, :]  # (B, L-1, V)
        target_ids = decoder_input_ids[:, 1:]  # (B, L-1)，shift one
                
        per_token_logps = []
        for logit_row, target_row in zip(logits, target_ids):
            log_probs = logit_row.log_softmax(dim=-1)  # (L-1, V)
            token_log_prob = torch.gather(log_probs, dim=1, index=target_row.unsqueeze(1)).squeeze(1)  # (L-1,)
            per_token_logps.append(token_log_prob)
                
        return torch.stack(per_token_logps)# (B, L-1)



    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_reward_func_name(self, reward_func) -> str:
        if isinstance(reward_func, PreTrainedModel):
            return reward_func.config._name_or_path.split("/")[-1]
        return reward_func.__name__

    def _aggregate_total_rewards(
        self,
        rewards_per_func: torch.Tensor,
        reward_func_names: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Default reward aggregation used by baseline NO-self-reward trainer."""
        del reward_func_names  # keep signature for subclass compatibility
        return rewards_per_func.sum(dim=1), {}

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        
        device = self.accelerator.device
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompts = []
        for item in inputs:
            if 'problem' in item:
                prompts.append(item['problem'])
            else:
                prompts.append('ERR')
        # print("[DEBUG] prompts_text 类型:", type(prompts_text))
        # print("[DEBUG] prompts_text 内容:", prompts_text)
                
        # Florence 接收 768*768尺寸的图像 所以这里直接写死为768*768即可
        images = []
        for x in inputs:
            if "image" in x:
                img = x["image"]
            elif "image_path" in x and x["image_path"] is not None:
                # print(f'[DEBUG] image_path is {x["image_path"]}')
                img = PIL.Image.open(x["image_path"])
                if img.mode != "RGB":
                    img = img.convert("RGB")


            img = img.resize((768, 768), PIL.Image.Resampling.LANCZOS)
            
            images.append(img)


        if len(images) > 0:
            prompt_inputs = self.processing_class(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        else:
            prompt_inputs = self.processing_class(
                text=prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # # resize output coordinate due to the image resize
        # origin_height = images[0].size[1]
        # origin_width = images[0].size[0]

        # option 1
        # resized_height, resized_width = smart_resize(origin_height, origin_width, max_pixels=self.processing_class.image_processor.max_pixels)
        # option 2
        # resized_height = prompt_inputs['image_grid_thw'][0][1] * self.processing_class.image_processor.patch_size
        # resized_width = prompt_inputs['image_grid_thw'][0][2] * self.processing_class.image_processor.patch_size
        
        # option3 for florence2 处理器可能 resize 后的图像尺寸
        pixel_values = prompt_inputs["pixel_values"]  # (B, C, H_resized, W_resized)
        _, _, resized_height, resized_width = pixel_values.shape
                
        # scale_x = origin_width / resized_width
        # scale_y = origin_height / resized_height
        # scales = [scale_x, scale_y]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        if len(images) > 0:
            pixel_values = prompt_inputs["pixel_values"]
            image_grid_thw =  [[1, resized_height, resized_width]]
        else:
            pixel_values = None
            image_grid_thw = None

        
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # Force eval mode for rollout generation.
            # Some Florence2 custom checkpoints branch on `self.training` in forward() and may touch
            # decoder2_input_ids even when it is None during generate, which crashes in train mode.
            was_training = unwrapped_model.training
            unwrapped_model.eval()
            try:
                # REFORM 的操作： generated_ids = model.module.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3)
                completion_ids = unwrapped_model.generate(
                    input_ids = prompt_inputs["input_ids"],
                    pixel_values = prompt_inputs["pixel_values"], 
                    inputs_embeds=None,
                    return_decoder2_outputs=False, 
                    For_RL = False,
                    max_new_tokens=1024,
                    num_beams=3,
                )
            finally:
                if was_training:
                    unwrapped_model.train()
            
            # No need to repeat prompt_mask as we're not duplicating prompts during generation

        # Mask everything after the first EOS token
        # is_eos = completion_ids == self.processing_class.eos_token_id
        
        ## 修改了completion_mask 的 计算逻辑，对florence2，将special_tokens mask掉即可

        special_tokens = torch.tensor([0, 1, 2, 3], device=completion_ids.device) # <pad>, <s>, </s>, <unk>
        completion_mask = (~torch.isin(completion_ids, special_tokens)).int()

        # print(f'[Debug] completion_mask:\n{completion_mask}')
        # print(f'[Debug] completion_mask sum per batch: {completion_mask.sum(dim=1)}')  # 每条序列有效长度
                

        # Concatenate prompt_mask with completion_mask for logit computation
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        pixel_values = prompt_inputs["pixel_values"]

            
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps_florence(
                    model, input_ids=prompt_ids, decoder_input_ids=completion_ids, attention_mask=completion_mask, pixel_values=pixel_values, 
                )
                # old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
                
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps_florence(
                    self.ref_model, input_ids=prompt_ids, decoder_input_ids=completion_ids,attention_mask=completion_mask, pixel_values=pixel_values, 
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps_florence(
                        model, input_ids=prompt_ids, decoder_input_ids=completion_ids, attention_mask=completion_mask, pixel_values=pixel_values, 
                    )
                    
        # florence2 的输出不包含 prompt 所以不需要裁剪         
        # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Decode the generated completions
        def clean_model_outputs_str(str_list, remove_tokens=None):
            """
            清理模型输出字符串列表，去掉无用 token。
            
            参数:
                str_list: list of str, 模型输出文本
                remove_tokens: list of str, 要移除的 token，默认 ['<s>', '</s>', '<pad>','unk']
                
            返回:
                cleaned_list: list of str，已清理文本
            """
            if remove_tokens is None:
                remove_tokens = ['<s>', '</s>', '<pad>', '<unk>']
            
            cleaned_list = []
            for s in str_list:
                content = s
                for token in remove_tokens:
                    content = content.replace(token, '')
                content = content.strip()
                cleaned_list.append(content)
            
            return cleaned_list
        
        completions = clean_model_outputs_str(self.processing_class.batch_decode(completion_ids, skip_special_tokens=False))
        
        ## 拼接进对话形式
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        reward_func_names = [self._get_reward_func_name(reward_func) for reward_func in self.reward_funcs]

        # Default: direct sum, can be overridden by subclasses (e.g., ARSPO grouped aggregation).
        rewards, extra_reward_logs = self._aggregate_total_rewards(rewards_per_func, reward_func_names)
        
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print(f'[Debug] in _generate_and_score_completions() advantages: {advantages}')
        
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func_name in enumerate(reward_func_names):
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        for metric_name, metric_value in extra_reward_logs.items():
            self._metrics[metric_name].append(float(metric_value))
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        # Check if we need to generate new completions or use buffered ones
        ## 当全局步骤 global_step 能被 num_iterations 整除时，表示进入了一个新的迭代周期，需要重新生成新的 inputs。否则，使用之前缓存的 inputs 来计算损失。
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs["pixel_values"]
        
        # # Concatenate for full sequence -- for Qwen
        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        
        ## 这里计算每个 token 的对数概率（log probability），这是基于 Florence2 模型的输出。
        # 具体来说，每个 token 的对数概率反映了模型在给定输入（包括图像和文本）情况下预测某个 token 的可能性。  
        per_token_logps = self._get_per_token_logps_florence(model=model, 
                                                             decoder_input_ids=completion_ids, 
                                                             input_ids=prompt_ids, 
                                                             attention_mask=completion_mask, 
                                                             pixel_values=pixel_values)
        # print(f'[Debug] per_token_logps is {per_token_logps}')
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        
        ## florence 2 不需要
        # per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]
        

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        # print(f'[Debug] coef_1 min/max: {coef_1.min().item()}, {coef_1.max().item()}')
        # print(f'[Debug] coef_2 min/max: {coef_2.min().item()}, {coef_2.max().item()}')

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # print(f'[Debug] per_token_loss1 min/max: {per_token_loss1.min().item()}, {per_token_loss1.max().item()}')
        # print(f'[Debug] per_token_loss2 min/max: {per_token_loss2.min().item()}, {per_token_loss2.max().item()}')
        # print(f'[Debug] per_token_loss min/max: {per_token_loss.min().item()}, {per_token_loss.max().item()}')
        

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            ## 因为在_get_per_token_logps_florence中获取 prob的时候，对competion做了-1操作，所以这里的mask 也得这样做
            mask_kl = completion_mask[:, 1:] 
            # print(f'[Debug] mask_kl sum per batch: {mask_kl.sum(dim=1)}')
            mean_kl = ((per_token_kl * mask_kl).sum(dim=1) / mask_kl.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * mask_kl).sum(dim=1) / mask_kl.sum(dim=1)).mean()
        # print(f'[Debug] final loss: {loss.item()}')
        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * mask_kl).sum() / mask_kl.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        # print("loss0:", loss)
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
