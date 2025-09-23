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

from .tasks import (
    TASK_MAP, 
    DataCollatorForAlpacaCLM
)
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
    parser.add_argument("--task", required=True, type=str, choices=list(TASK_MAP.keys()))
    parser.add_argument("--lora_adapter", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()


    # load model  
    model = AutoModelForCausalLM.from_pretrained(args.hf_name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_name_or_path)


    # load dataset
    if args.task == "wikitext":
        TaskClass = TASK_MAP[args.task]    
        task = TaskClass(splits=["train", "validation", "test"], 
                        load_from_disk=False, #TODO
                        local_dir=None) #TODO
        train_dataloader, val_dataloader = task.get_dataloaders(
            list_splits=["train", "validation"], 
            batch_size=args.batch_size)
        preprocess_fn = None
    elif args.task == "alpaca_plus": 
        TaskClass = TASK_MAP[args.task] 
        task = TaskClass(splits=["train", "validation_seen", "validation_unseen", "validation_human"], 
                         load_from_disk=False, #TODO
                         local_dir=None) #TODO
        train_dataloader, val_dataloader = task.get_dataloaders(
            list_splits=["train", "validation_unseen"], 
            batch_size=args.batch_size)

        # tokenizer expand vocab for gist token

        tokenizer.add_special_tokens({"additional_special_tokens":["<GIST>"]})
        model.resize_token_embeddings(len(tokenizer))
        gist_token_id = tokenizer.convert_tokens_to_ids("<GIST>")
        # initialize word embeddings by averaging the embeddings
        with torch.no_grad():
            model.model.embed_tokens.weight[gist_token_id] = model.model.embed_tokens.weight[:gist_token_id].mean(dim=0)
            model.lm_head.weight[gist_token_id] = model.lm_head.weight[:gist_token_id].mean(dim=0)


        preprocess_fn = DataCollatorForAlpacaCLM(tokenizer=tokenizer,
                                                 max_length=512,
                                                 max_length_human=512, 
                                                 label_pad_token_id=-100,
                                                 return_tensors="pt",
                                                 gist_token=50257, 
                                                 pad_token=0, 
                                                 add_gist_token=True, 
                                                #  gist_condition="gist", 
                                                 num_gist_tokens=1, 
                                                 check_correctness=False)


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
        preprocess_fn=preprocess_fn,
        args=args, 
    )
    trainer.train()