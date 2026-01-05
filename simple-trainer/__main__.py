import os 
from argparse import ArgumentParser, Namespace 
from copy import deepcopy 
from typing import Tuple, List, Dict, Union, Self, Optional
import dataclasses

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributed.fsdp import MixedPrecision
# import torchao
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    get_peft_model, 
    LoraConfig, 
    LoraModel, 
    PeftConfig, 
    TaskType
)
from accelerate import Accelerator 
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import (
    AORecipeKwargs
)

from .tasks import (
    TASK_MAP, 
    COMPRESSION_TASKS, 
    DataCollatorForAlpacaCLM, 
    DataCollatorForAlpacaPlusPlusCLM
)
from .trainer import SimpleTrainer, MICompressTrainer, MICompressTrainerWithCoordinateDescent
from .compress_utils import get_compression_model, get_mi_compression_model
from .metrics import get_compute_metrics_fn


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

if int(os.environ["DEBUGPY_ENABLE"]) == 1:
    import debugpy 
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        debugpy.listen(("172.26.93.56", 5678 + local_rank))
        debugpy.wait_for_client()


if __name__ == "__main__": 

    parser = ArgumentParser("Simple Trainer")
    parser.add_argument("--hf_name_or_path", type=str, required=True, help="Huggingface identifier or path")
    parser.add_argument("--task", required=True, type=str, choices=list(TASK_MAP.keys()))
    parser.add_argument("--lora_adapter", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./runs")
    parser.add_argument("--num_gist_tokens", type=int, default=1)
    parser.add_argument("--turn_off_gist_masking", action="store_true")
    # parser.add_argument("--num_compress_steps", type=int, default=1800, help="number of steps to learn language modeling objective before learning the critic")
    # parser.add_argument("--num_critic_steps", type=int, default=200, help="number of steps to learn the critic for maximizing MI")
    parser.add_argument("--do_coordinate_descent", action="store_true")
    parser.add_argument("--compress_fraction", default=0.25, type=float)
    parser.add_argument("--alpha", type=float, default=1e-2, help="DV loss mixing ratio")
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
        
        test_dataloaders: Optional[List[DataLoader]] = None
        test_names = None

        preprocess_fn = None
        compute_metrics_fn = None
        optimizing_metric = "loss"
        TrainerClass = SimpleTrainer

    elif args.task == "alpaca_plus": 
        TaskClass = TASK_MAP[args.task] 
        task = TaskClass(splits=["train", "validation_seen", "validation_unseen", "validation_human"], 
                         load_from_disk=False, #TODO
                         local_dir=None) #TODO
        train_dataloader, val_dataloader = task.get_dataloaders(
            list_splits=["train", "validation_unseen"], 
            batch_size=args.batch_size)

        test_dataloaders: Optional[List[DataLoader]] = task.get_dataloaders(
            list_splits=["validation_seen", "validation_unseen", "validation_human"]
        )
        test_names = ["validation_seen", "validation_unseen", "validation_human"]


        model, tokenizer = get_compression_model(model, tokenizer, args.turn_off_gist_masking)
        tokenizer.padding_side = "left"


        tokenizer.pad_token_id = tokenizer.eos_token_id
        preprocess_fn = DataCollatorForAlpacaCLM(tokenizer=tokenizer,
                                                 max_length=512,
                                                 max_length_human=768, 
                                                 label_pad_token_id=-100,
                                                 return_tensors="pt",
                                                 gist_token=model.gist_token_id, 
                                                 pad_token=tokenizer.pad_token_id, 
                                                 add_gist_token=True, 
                                                 num_gist_tokens=args.num_gist_tokens,  
                                                 check_correctness=False)
        compute_metrics_fn = get_compute_metrics_fn(model.gist_token_id, tokenizer, args)
        optimizing_metric = "rougeL"
        TrainerClass = SimpleTrainer

    elif args.task == "alpaca_pp": 
        TaskClass = TASK_MAP[args.task] 
        task = TaskClass(splits=["train", "validation_seen", "validation_unseen", "validation_human"], 
                         load_from_disk=False, #TODO
                         local_dir=None) #TODO
        train_dataloader, val_dataloader = task.get_dataloaders(
            list_splits=["train", "validation_unseen"], 
            batch_size=args.batch_size)

        test_dataloaders: Optional[List[DataLoader]] = task.get_dataloaders(
            list_splits=["validation_seen", "validation_unseen", "validation_human"]
        )
        test_names = ["validation_seen", "validation_unseen", "validation_human"]

        #TODO: get_mi_compression_model
        model, tokenizer = get_mi_compression_model(model, tokenizer, args.turn_off_gist_masking) 
        tokenizer.padding_side = "left"


        tokenizer.pad_token_id = tokenizer.eos_token_id
        preprocess_fn = DataCollatorForAlpacaPlusPlusCLM(tokenizer=tokenizer,
                                                 max_length=512,
                                                 max_length_human=768, 
                                                 label_pad_token_id=-100,
                                                 return_tensors="pt",
                                                 gist_token=model.gist_token_id, 
                                                 pad_token=tokenizer.pad_token_id, 
                                                 add_gist_token=True, 
                                                 num_gist_tokens=args.num_gist_tokens,  
                                                 check_correctness=False)
        compute_metrics_fn = get_compute_metrics_fn(model.gist_token_id, tokenizer, args)
        optimizing_metric = "rougeL"
        TrainerClass = MICompressTrainerWithCoordinateDescent if args.do_coordinate_descent else MICompressTrainer
    else: 
        train_dataloader, val_dataloader = None, None 
        test_dataloaders: Optional[List[DataLoader]] = None 
        test_names = None

        preprocess_fn = None 
        compute_metrics_fn = None 
        optimizing_metric = None
        TrainerClass = SimpleTrainer

        raise NotImplementedError 

    if args.lora_adapter is not None: 
        if args.lora_adapter == "random_init": 
            pass
            #TODO: initialize an empty lora adapter
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=256, 
                target_modules="all-linear", 
                lora_alpha=32, 
                init_lora_weights="gaussian"
            )
            # model = LoraModel(model, lora_config, adapter_name="default")
            # model = get_peft_model(model, lora_config)
            model.add_adapter(lora_config, "default")
            if args.task in COMPRESSION_TASKS: 
                # pass
                model.unlock_embeddings()
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
    iters = args.epochs * len(train_dataloader) / args.batch_size 
    lr_sched = CosineAnnealingLR(optim, T_max=iters)



    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)


    accel = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        step_scheduler_with_optimizer=False, 
        kwargs_handlers=[ddp_kwargs]
    )

    train_dataloader, model, optim, lr_sched, val_dataloader = accel.prepare(train_dataloader, model, optim, lr_sched, val_dataloader)
    if test_dataloaders is not None: 
        test_dataloaders = [accel.prepare(t) for t in test_dataloaders]


    # if args.alpha > 0 and args.task in ["alpaca_pp"]: 
    #     config = accel.unwrap_model(model).config
    #     critic = nn.Sequential( 
    #         nn.Linear(config.hidden_size, 1, bias=False), 
    #     ).to("cuda")
    #     model.module.add_critic_network(critic)
    #     print("\nCritic Network Added\n")
    #     optim.add_param_group({"params": critic.parameters(), "lr": args.lr * 1e-3, "weight_decay":1e0})
    # else: 
    #     raise ValueError

    accel.wait_for_everyone() 

    trainer = TrainerClass(
        model, tokenizer, optim, lr_sched, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        accelerator=accel, 
        preprocess_fn=preprocess_fn,
        compute_metrics=compute_metrics_fn, 
        output_dir=args.output_dir, 
        optimizing_metric=optimizing_metric, 
        args=args, 
    )

    trainer.train()

    trainer.load_best_checkpoint()

    if test_dataloaders is not None:
        for name, dl in zip(test_names, test_dataloaders): 
            preds = trainer.evaluate(dataloader=dl, generate=True)
            mets = compute_metrics_fn(preds)
            mets["eval_loss"] = preds.losses.mean().item()
            # mets = None
            if accel.is_main_process:
                print(f"===== EVAL RESULTS: {name} =====")
                print(mets)
