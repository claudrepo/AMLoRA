import os
import sys
import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union, Dict

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
# 添加masked_evaluator_implementation.py的路径
sys.path.append(os.path.join(os.getcwd(), '..'))
print("添加的masked_evaluator路径:", os.path.join(os.getcwd(), '..'))

# 检查是否能找到masked_evaluator_implementation.py
masked_evaluator_path = os.path.join(os.getcwd(), '..', 'masked_evaluator_implementation.py')
print(f"masked_evaluator_implementation.py存在吗? {os.path.exists(masked_evaluator_path)}")

# 尝试导入掩码评估器相关类
try:
    from masked_evaluator_implementation import MaskedAMLoRAConfig, MaskedAMLoRALayer, MaskedEvaluatorManager, MaskedEvaluatorLayer
    HAS_MASKED_MIXER = True
    print("成功导入掩码评估器实现")
except ImportError as e:
    print(f"警告：无法导入masked_evaluator_implementation.py中的类，将使用标准AMLoRA实现: {e}")
    HAS_MASKED_MIXER = False

# 自定义Trainer类来集成掩码评估器功能
if HAS_MASKED_MIXER:
    class MaskedAMLoRATrainer(transformers.Trainer):
        def training_step(self, model, inputs):
            # 标准的训练步骤
            loss = super().training_step(model, inputs)
            
            # 如果启用了掩码评估器，更新掩码
            if hasattr(model, 'use_masked_evaluator') and model.use_masked_evaluator:
                # 检查是否需要更新掩码
                if model.evaluator_manager.should_update_mask(self.state.global_step):
                    # 获取当前预算（传入全局步骤以确保正确的warmup行为）
                    current_budget = model.evaluator_manager.get_current_budget(self.state.global_step)
                    
                    # 更新掩码
                    for name, module in model.named_modules():
                        if isinstance(module, MaskedAMLoRALayer):
                            module.update_evaluator_mask(current_budget, self.state.global_step)
                    
                    # 计算预算比例（而不是预算数量）
                    total_params = model.evaluator_manager.total_params
                    budget_ratio = current_budget / total_params
                    
                    # 记录当前的掩码预算比例
                    self.log({
                        "mask_budget_ratio": budget_ratio,
                        "budget_count": current_budget,
                        "step": self.state.global_step
                    })
                
                # 更新预算调度
                model.evaluator_manager.step()
                
            return loss

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        use_amlora: bool=False,
        # 掩码评估器参数
        enable_masked_evaluator: bool = False,
        initial_mask_budget_ratio: float = 1.0,
        final_mask_budget_ratio: float = 0.5,
        budget_schedule_steps: int = 10000,
        mask_update_frequency: int = 100,
        evaluator_beta1: float = 0.85,
        evaluator_beta2: float = 0.85,
        warmup_steps: int = 0,  # warmup阶段步数，在该阶段保持initial_mask_budget_ratio不变
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        use_gradient_checkpointing: bool = False,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"use_amlora: {use_amlora}\n" #! added
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"enable_masked_evaluator: {enable_masked_evaluator}\n"
        f"initial_mask_budget_ratio: {initial_mask_budget_ratio}\n"
        f"final_mask_budget_ratio: {final_mask_budget_ratio}\n"
        f"budget_schedule_steps: {budget_schedule_steps}\n"
        f"mask_update_frequency: {mask_update_frequency}\n"
        f"evaluator_beta1: {evaluator_beta1}\n"
        f"evaluator_beta2: {evaluator_beta2}\n"
        f"warmup_steps: {warmup_steps}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
            local_files_only=True,
        )

    if "llama2" in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if adapter_name == "lora":
        if enable_masked_evaluator and HAS_MASKED_MIXER:
            print("启用掩码评估器功能")
            # 创建掩码AMLoRA配置
            masked_config = MaskedAMLoRAConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_use_evaluator=use_amlora,
                initial_mask_budget_ratio=initial_mask_budget_ratio,
                final_mask_budget_ratio=final_mask_budget_ratio,
                budget_schedule_steps=budget_schedule_steps,
                mask_update_frequency=mask_update_frequency,
                evaluator_beta1=evaluator_beta1,
                evaluator_beta2=evaluator_beta2,
                warmup_steps=warmup_steps
            )
            
            # 修改PEFT配置以使用掩码评估器
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_use_evaluator=use_amlora,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # 这里我们使用标准的get_peft_model，但会在后面的训练循环中添加掩码评估器的更新逻辑
            model = get_peft_model(model, config)
            
            # 记录我们启用了掩码评估器
            model.use_masked_evaluator = True
            model.masked_config = masked_config
            model.evaluator_manager = MaskedEvaluatorManager(masked_config)
        else:
            if enable_masked_evaluator:
                print("警告：无法启用掩码评估器，因为缺少相关实现")
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_use_evaluator=use_amlora, #! added
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            model.use_masked_evaluator = False
    if adapter_name == "prefix-tuning":
        model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check if resume_from_checkpoint is a valid path before trying to load
        if os.path.exists(resume_from_checkpoint):
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                model = set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
        else:
            print(f"警告：resume_from_checkpoint路径不存在: {resume_from_checkpoint}")
            print("将开始全新训练，不加载任何检查点")
            resume_from_checkpoint = None  # 重置为None，避免后续错误

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # 根据是否启用掩码评估器选择合适的Trainer类
    if HAS_MASKED_MIXER and hasattr(model, 'use_masked_evaluator') and model.use_masked_evaluator:
        TrainerClass = MaskedAMLoRATrainer
        print("使用自定义MaskedAMLoRATrainer")
    else:
        TrainerClass = transformers.Trainer
        print("使用标准Trainer")
    
    trainer = TrainerClass(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501


if __name__ == "__main__":
    fire.Fire(train)
