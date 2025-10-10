import os 
import sys 
from typing import List, Dict, Tuple, Optional, Union, Self
from time import sleep
import multiprocessing
import tqdm
import logging
from argparse import Namespace
import math
from tqdm_multiprocess.logger import setup_logger_tqdm
import shutil

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer, 
    EvalPrediction
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
                 preprocess_fn: Optional[callable] = None, 
                 compute_metrics: Optional[callable] = None, 
                 optimizing_metric: Optional[str] = None): 
        self.model = model 
        self.tokenizer = tokenizer
        self.optim = optim 
        self.lr_scheduler = lr_sched 
        self.train_dloader = train_dataloader
        self.val_dloader = val_dataloader 
        self.accel = accelerator
        self.args = args

        self.preprocess_fn = DEFAULT_CAUSAL_LM_PREPROCESS_FN if preprocess_fn is None else preprocess_fn
        self.compute_metrics = compute_metrics
        
        self.current_step = 0
        self.epochs = self.args.epochs 
        self.batch_size = self.args.batch_size
        self.eval_steps = self.args.eval_steps
        self.optimizing_metric = "loss" if optimizing_metric is None else optimizing_metric 
        self.best_metric = -np.inf if optimizing_metric != "loss" else np.inf

        self.output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir 
        os.makedirs(self.output_dir, exist_ok=True)

        self.all_mets_file = os.path.join(self.output_dir, "all_metrics.log")
        self.best_mets_file = os.path.join(self.output_dir, "best_metrics.log")
        if os.path.exists(self.all_mets_file): 
            shutil.rmtree(self.all_mets_file)
        if os.path.exists(self.best_mets_file): 
            shutil.rmtree(self.best_mets_file)

        self.save_checkpoint(save_dir=os.path.join(self.output_dir, "best_tfmr"))


    def train(self): 

        num_steps_to_train = self.epochs * len(self.train_dloader) 
        cur_epoch = 0
        
        tbar = tqdm.tqdm(range(num_steps_to_train), desc=f"Training {cur_epoch}/{self.epochs} Epochs", total=num_steps_to_train)
        iter_train_dloader = iter(self.train_dloader) 
        self.model.train()
        for i in tbar: 
            self.model.train()
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
                                            labels=batch["labels"], 
                                            attention_mask=batch["attention_mask"], 
                                            use_cache=False)
                    logits = outputs.logits
                    loss = outputs.loss
                    # loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1))

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
            if i%self.eval_steps == 0 and i > 0: 
                eval_preds = self.evaluate(generate=True)

                # DEBUG 
                # os.makedirs(os.path.join(self.output_dir, "preds"), exist_ok=True)
                # if self.accel.is_main_process: 
                #     print(eval_preds.predictions[:2])
                # with open(os.path.join(self.output_dir, "preds", f"{self.accel.process_index}.log"), "w") as f: 
                #     [f.write(f"{self.tokenizer.decode(torch.where(p == -100, self.tokenizer.pad_token_id, p), skip_special_tokens=True)}\n") for p in eval_preds.predictions]



                metrics = {}
                if self.compute_metrics is not None: 
                    metrics = self.compute_metrics(eval_preds)
                    metrics["eval_loss"] = eval_preds.losses.mean().item()
                else: 
                    metrics = {"eval_loss": eval_preds.losses.mean().item()}

                if self.accel.is_main_process:
                    print("\n\n", metrics)
                
                # save model if best metric has been obtained
                higher_is_better = self.optimizing_metric != "loss" 
                if higher_is_better: 
                    operator = np.greater 
                else:
                    operator = np.less 

                if self.accel.is_main_process: 
                    with open(os.path.join(self.output_dir, "all_metrics.log"), "a") as f: 
                        f.write(f"step:{i},")
                        f.write(",".join([f"{k}:{v}" for k, v in metrics.items()]))
                        f.write("\n")

                if operator(metrics[self.optimizing_metric], self.best_metric): 
                    self.best_metric = metrics[self.optimizing_metric] 
                    self.save_checkpoint(save_dir = os.path.join(self.output_dir, "best_tfmr"))

                    if self.accel.is_main_process: 
                        with open(os.path.join(self.output_dir, "best_metrics.log"), "a") as f: 
                            f.write(f"step:{i},")
                            f.write(",".join([f"{k}:{v}" for k, v in metrics.items()]))
                            f.write("\n")
                


    def evaluate(self, generate=False) -> EvalPrediction:
        self.model.eval()

        iter_val_loader = iter(self.val_dloader) 
        num_val_steps = len(self.val_dloader)

        vbar = tqdm.tqdm(range(num_val_steps), desc="Validating")

        for step in vbar: 
            try:
                batch = next(iter_val_loader)
            except StopIteration: 
                break 
            
            batch = self.preprocess_fn(batch, self.tokenizer, use_chat_template=False).to(self.accel.device)

            preds = None 
            labels = None
            losses = None

            all_preds = None 
            all_labels = None
            all_losses = None

            with torch.no_grad():
                outputs = self.model(input_ids=batch["input_ids"], 
                                    labels=batch["labels"], 
                                    attention_mask=batch["attention_mask"], 
                                    use_cache=False)
                logits = outputs.logits
                loss = outputs.loss
                # loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1)) 

                losses = self.accel.gather(loss) 



                if generate: 
                    if "labels" in batch:
                        labels = batch["labels"]
                        input_ids = batch["prompt_input_ids"] 
                        attention_mask = batch["prompt_attention_mask"]
                    else:
                        labels = None 
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                    

                    gen_outputs = self.model.generate(input_ids=input_ids, 
                                                    attention_mask=attention_mask, 
                                                    use_cache=True, return_dict_in_generate=True, 
                                                    max_new_tokens=512)
                    preds = gen_outputs.sequences
                    if self.accel.is_main_process:
                        print("INPUTS:", self.tokenizer.batch_decode(batch["input_ids"])[0])
                        print("LABELS:", self.tokenizer.batch_decode(torch.where(labels == -100, self.tokenizer.pad_token_id, labels))[0])
                        print("PREDS:", self.tokenizer.batch_decode(preds)[0])

                    # pad and gather predictions 
                    preds = self.accel.pad_across_processes(preds, 
                                                            dim=-1, 
                                                            pad_index=-100,
                                                            pad_first=self.tokenizer.padding_side=="left")
                    preds = self.accel.gather(preds)
                    # pad and gather labels
                    if labels is not None: 
                        labels = self.accel.pad_across_processes(labels, 
                                                                 dim=-1, 
                                                                 pad_index=-100, 
                                                                 pad_first=self.tokenizer.padding_side=="left")
                        labels = self.accel.gather(labels)
                    
                    all_preds = (preds.cpu() if all_preds is None else torch_pad_and_concatenate(all_preds, preds.cpu(), padding_index=-100))
                    all_labels = (labels.cpu() if all_labels is None else torch_pad_and_concatenate(all_preds, preds.cpu(), padding_index=-100))
                    all_losses = (losses.cpu() if all_losses is None else torch.cat([all_losses, losses.cpu()]))
            

            vbar.set_postfix({"eval_loss": loss.item()})
            vbar.set_description(f"Validating {step+1}/{num_val_steps} steps")


        return EvalPrediction(
            predictions=all_preds, 
            label_ids=all_labels, 
            losses=all_losses
        )
        
    def save_checkpoint(self, save_dir=None): 
        if save_dir is None: 
            save_dir = self.output_dir 
        # save model 
        unwrapped_model = self.accel.unwrap_model(self.model) 
        unwrapped_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir) 



 

# From huggingface
def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = torch.atleast_1d(tensor1)
    tensor2 = torch.atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    # Now let's fill the result tensor
    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result
    
        


if __name__ == "__main__": 

    from .tasks import WikitextTask 

    wikitext = WikitextTask()
    train_dloader = wikitext.get_dataloaders(["train"])[0]
    batch  = next(iter(train_dloader))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 
    
    output = DEFAULT_CAUSAL_LM_PREPROCESS_FN(batch, tok) 

