import os 
from argparse import ArgumentParser, Namespace 
from copy import deepcopy 
from typing import Tuple, List, Dict, Union, Self, Optional
import dataclasses

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    get_peft_model, 
    LoraConfig, 
    LoraModel, 
    PeftConfig, 
    TaskType
)
from accelerate import Accelerator 

from .tasks import TASK_MAP
from .trainer import SimpleTrainer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


if __name__ == "__main__": 

    parser = ArgumentParser("Simple Trainer")
    parser.add_argument("--hf_name_or_path", type=str, required=True, help="Huggingface identifier or path")
    parser.add_argument("--task", required=True)
    parser.add_argument("--lora_adapter", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    # load dataset
    TaskClass = TASK_MAP[args.task]    
    task = TaskClass(splits=["train", "validation", "test"], 
                     load_from_disk=False, #TODO
                     local_dir=None) #TODO

    train_dataloader, val_dataloader = task.get_dataloaders(
        list_splits=["train", "validation"], 
        batch_size=args.batch_size)

    # load model  
    model = AutoModelForCausalLM.from_pretrained(args.hf_name_or_path, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_name_or_path)

    if args.lora_adapter is not None: 
        if args.lora_adapter == "random_init": 
            pass
            #TODO: initialize an empty lora adapter
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=64, 
                target_modules="all-linear", 
                lora_alpha=32, 
                init_lora_weights="gaussian"
            )
            # model = LoraModel(model, lora_config, adapter_name="default")
            model = get_peft_model(model, lora_config)
        else: 
            pass
            #TODO: load lora_adapter with defined name 

    print_trainable_parameters(model)

    # optimizer assignment 
    if isinstance(model, LoraModel): 
        params = filter(lambda p:p.requires_grad, model.parameters())
    else: 
        params = model.parameters()
    optim = AdamW(params, lr=args.lr)
    iters = args.epochs * len(train_dataloader) / args.batch_size /args.gradient_accumulation_steps
    lr_sched = CosineAnnealingLR(optim, T_max=iters)

    accel = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        mixed_precision="bf16")
    train_dataloader, model, optim, lr_sched = accel.prepare(train_dataloader, model, optim, lr_sched)

    trainer = SimpleTrainer(
        model, tokenizer, optim, lr_sched, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        accelerator=accel, 
        args=args, 
    )
    trainer.train()