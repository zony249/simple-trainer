import os 
from argparse import ArgumentParser, Namespace 
from copy import deepcopy 
from typing import Tuple, List, Dict, Union, Self, Optional
import dataclasses

import torch

from transformers import AutoModelForCausalLM
from peft import (
    PeftConfig, 
    get_peft_model
)
from accelerate import Accelerator 

from .tasks import TASK_MAP

if __name__ == "__main__": 

    parser = ArgumentParser("Simple Trainer")
    parser.add_argument("--hf_name_or_path", type=str, required=True, help="Huggingface identifier or path")
    parser.add_argument("--task", required=True)
    parser.add_argument("--lora_adapter", type=str, default=None)

    args = parser.parse_args()

    # load dataset
    TaskClass = TASK_MAP[args.task]    
    task = TaskClass(splits=["train", "validation", "test"], 
                     load_from_disk=False, #TODO
                     local_dir=None) #TODO

    trainset, valset = task.get_splits(["train", "validation"])

    # load model  
    model = AutoModelForCausalLM.from_pretrained(args.hf_name_or_path)


    if args.lora_adapter is not None: 
        if args.lora_adapter == "random_init": 
            pass
            #TODO: initialize an empty lora adapter
        else: 
            pass
            #TODO: load lora_adapter with defined name 

    # accel = Accelerator()

    