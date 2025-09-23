import os 
import sys 
from typing import List, Dict, Tuple, Optional, Union, Self
from time import sleep
import multiprocessing
import tqdm
import logging
from argparse import Namespace
from tqdm_multiprocess.logger import setup_logger_tqdm

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer
)

from accelerate import Accelerator


# Globals 
DEFAULT_OUTPUT_DIR = "./runs"

def DEFAULT_CAUSAL_LM_PREPROCESS_FN(batch: List[Union[Tuple[str, str], str]], 
                                    tokenizer: PreTrainedTokenizer,
                                    use_chat_template: Optional[bool] = False): 
    """
    Performs batching, label shifting, and applies chat template.

    ### Batch Formatting (IMPORTANT)
    if use_chat_template, then expect batch = [(user, assistant), ..., ]
    otherwise, expect batch = [text1, text2, ...]
    """
    if use_chat_template: 
        assert isinstance(batch[0], tuple) and len(batch[0]) == 2, \
            f"use_chat_template is True, there expect batch to be List of Tuple(user: str, assistant: str)."
        preproc_batch = []
        for item in batch:
            chat = [{"role": "user", "content": item[0]}, 
                    {"role": "assistant", "content": item[1]}]
            chat_t = tokenizer.apply_chat_template(chat, tokenize=False)
            preproc_batch.append(chat_t)
    else: 
        assert isinstance(batch[0], str), \
            f"For causal language modelling objective, expect batch to be List[str]."
        preproc_batch = batch

    if tokenizer.pad_token_id is None: 
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_batch = tokenizer(preproc_batch, 
                            truncation=True, 
                            padding=True,
                            max_length=1024, 
                            return_tensors="pt")
    # shift tokens 
    labels = torch.cat([tokenized_batch["input_ids"][:, 1:], 
                        tokenizer.pad_token_id * torch.ones((len(batch), 1), dtype=torch.long)], dim=-1)

    tokenized_batch["labels"] = labels 
    tokenized_batch["attention_mask"][:, -1] = 0

    return tokenized_batch

logger = logging.getLogger(__name__)



class SimpleTrainer: 
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer, 
                 optim : Optimizer, 
                 lr_sched : LRScheduler, 
                 train_dataloader : DataLoader, 
                 val_dataloader : DataLoader, 
                 accelerator: Accelerator, 
                 args: Namespace, 
                 output_dir : Optional[str] = None, 
                 preprocess_fn: Optional[callable] = None): 
        self.model = model 
        self.tokenizer = tokenizer
        self.optim = optim 
        self.lr_scheduler = lr_sched 
        self.train_dloader = train_dataloader
        self.val_dloader = val_dataloader 
        self.accel = accelerator
        self.args = args

        self.output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir 
        self.preprocess_fn = DEFAULT_CAUSAL_LM_PREPROCESS_FN if preprocess_fn is None else preprocess_fn
        
        self.current_step = 0
        self.epochs = self.args.epochs 
        self.batch_size = self.args.batch_size

    def train(self): 

        num_steps_to_train = self.epochs * len(self.train_dloader) 
        cur_epoch = 0
        
        tbar = tqdm.tqdm(range(num_steps_to_train), desc=f"Training {cur_epoch}/{self.epochs} Epochs", total=num_steps_to_train)
        iter_train_dloader = iter(self.train_dloader) 
        self.model.train()
        for i in tbar: 

            # Make dataloader iterator repeatable.
            try: 
                batch = next(iter_train_dloader) 
            except StopIteration: 
                iter_train_dloader = iter(self.train_dloader) 
                batch = next(iter_train_dloader) 
            
            batch = self.preprocess_fn(batch, self.tokenizer, use_chat_template=False).to(self.accel.device)    

            with self.accel.accumulate(self.model):
                with self.accel.autocast():

                    outputs = self.model(input_ids=batch["input_ids"],
                                            attention_mask=batch["attention_mask"], 
                                            use_cache=False)
                    logits = outputs.logits
                    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1))

                self.accel.backward(loss)
                # loss.backward()
                self.optim.step() 
                self.lr_scheduler.step()
                self.optim.zero_grad()


            # update tbar desc
            cur_epoch = i / num_steps_to_train * self.epochs 
            tbar.set_description(f"Training {cur_epoch:.2f}/{self.epochs} Epochs")
            # update tbar postfix
            tbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.lr_scheduler.get_last_lr()[0]:.4e}"})




            pass 
        


if __name__ == "__main__": 

    from .tasks import WikitextTask 

    wikitext = WikitextTask()
    train_dloader = wikitext.get_dataloaders(["train"])[0]
    batch  = next(iter(train_dloader))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 
    
    output = DEFAULT_CAUSAL_LM_PREPROCESS_FN(batch, tok) 

