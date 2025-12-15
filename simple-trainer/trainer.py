import os 
import sys 
from typing import List, Dict, Tuple, Optional, Union, Self, Iterable
from time import sleep
import multiprocessing
import tqdm
import logging
from argparse import Namespace
import math
from tqdm_multiprocess.logger import setup_logger_tqdm
import shutil
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    PreTrainedModel, 
    PreTrainedTokenizer, 
    EvalPrediction, 
    BatchEncoding
)

from accelerate import Accelerator, load_checkpoint_and_dispatch, DistributedDataParallelKwargs
from accelerate.utils import (
    merge_fsdp_weights
)
from peft import PeftModel
from .models.utils import get_first_idx_of_token, get_last_idx_of_token


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
        # if isinstance(self.accel.unwrap_model(self.model), PeftModel): 
        #     best_path = os.path.join(self.output_dir, "best_tfmr")


        self.__post_init__()

    def __post_init__(self): 
        print("========= SIMPLE TRAINER ==========")

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

                    loss, loss_mets = self.compute_loss(batch, return_loss_breakdown=True)
                    # loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1))

                self.accel.backward(loss)
                loss = loss.detach() 

                # loss.backward()
                self.optim.step() 
                self.lr_scheduler.step()
                self.optim.zero_grad()
                # torch.cuda.empty_cache()

            self.accel.wait_for_everyone()

            # update tbar desc
            cur_epoch = i / num_steps_to_train * self.epochs 
            tbar.set_description(f"Training {cur_epoch:.2f}/{self.epochs} Epochs")
            # update tbar postfix
            if len(loss_mets) > 0:
                loss_mets_stringify = {k: f"{v:.4f}" for k, v in loss_mets.items()} 
                tbar.set_postfix(loss_mets_stringify | {"lr": f"{self.lr_scheduler.get_last_lr()[0]:.4e}"})
            else: 
                tbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.lr_scheduler.get_last_lr()[0]:.4e}"})
            if i%self.eval_steps == 0 and i > 0: 
                eval_preds = self.evaluate(generate=True)

                # DEBUG 
                os.makedirs(os.path.join(self.output_dir, "preds"), exist_ok=True)
                if self.accel.is_main_process: 
                    print(eval_preds.predictions[:2])
                with open(os.path.join(self.output_dir, "preds", f"{self.accel.process_index}.log"), "w") as f: 
                    [f.write(repr(f"{self.tokenizer.decode(torch.where(p == -100, self.tokenizer.pad_token_id, p), skip_special_tokens=True)}") + "\n") for p in eval_preds.predictions]

                os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
                if self.accel.is_main_process: 
                    print(eval_preds.label_ids[:2])
                with open(os.path.join(self.output_dir, "labels", f"{self.accel.process_index}.log"), "w") as f: 
                    [f.write(repr(f"{self.tokenizer.decode(torch.where(p == -100, self.tokenizer.pad_token_id, p), skip_special_tokens=True)}") + "\n") for p in eval_preds.label_ids]


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
                
    def compute_loss(self, batch: BatchEncoding, 
                     return_loss_breakdown=True) -> torch.FloatTensor: 
        outputs = self.model(input_ids=batch["input_ids"],
                                labels=batch["labels"], 
                                attention_mask=batch["attention_mask"], 
                                use_cache=False)
        
        logits = outputs.logits
        loss = outputs.loss

        if return_loss_breakdown: 
            mets = {"loss": loss.item()}
            return loss, mets
        return loss


    def evaluate(self, 
                 dataloader: Optional[DataLoader] = None, 
                 generate: Optional[bool] = False) -> EvalPrediction:
        self.model.eval()

        model = self.accel.unwrap_model(self.model)
        # model = self.model


        iter_val_loader = iter(self.val_dloader if dataloader is None else dataloader) 
        num_val_steps = len(self.val_dloader if dataloader is None else dataloader)

        vbar = tqdm.tqdm(range(num_val_steps), desc="Validating")

        all_preds = None 
        all_labels = None
        all_losses = None

        for step in vbar: 
            try:
                batch = next(iter_val_loader)
            except StopIteration: 
                break 
            
            batch = self.preprocess_fn(batch, self.tokenizer, use_chat_template=False).to(self.accel.device)

            preds = None 
            labels = None
            losses = None


            with torch.no_grad():
                self.accel.wait_for_everyone()
                outputs = model(input_ids=batch["input_ids"], 
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
                        input_ids = batch["prompt_input_ids"] #if "prompt_input_ids" in batch else batch["input_ids"]
                        attention_mask = batch["prompt_attention_mask"] #if "prompt_attention_mask" in batch else batch["attention_mask"]
                    else:
                        labels = None 
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                    

                    gen_outputs = model.generate(input_ids=input_ids, 
                                                    attention_mask=attention_mask, 
                                                    use_cache=True, return_dict_in_generate=True, 
                                                    max_new_tokens=512)
                    preds = gen_outputs.sequences

                    self.accel.wait_for_everyone()

                    # The prompt is included in the generated tokens. Remove this.
                    assert (
                        preds[:, : input_ids.shape[-1]] == input_ids
                    ).all()
                    preds = preds[:, input_ids.shape[-1] :]

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
                    all_labels = (labels.cpu() if all_labels is None else torch_pad_and_concatenate(all_labels, labels.cpu(), padding_index=-100))
                    all_losses = (losses.cpu() if all_losses is None else torch.cat([all_losses, losses.cpu()]))
            

            vbar.set_postfix({"eval_loss": loss.item()})
            vbar.set_description(f"Validating {step+1}/{num_val_steps} steps")

            self.accel.wait_for_everyone()
            # break # DEBUG

        print(f"Number of samples predicted: {all_preds.shape}")
        print(f"Number of references: {all_labels.shape}")

        return EvalPrediction(
            predictions=all_preds, 
            label_ids=all_labels, 
            losses=all_losses
        )
        
    def save_checkpoint(self, save_dir=None): 
        if save_dir is None: 
            save_dir = self.output_dir 
        # save model 
        self.tokenizer.save_pretrained(save_dir) 

        if self.accel.mixed_precision == "fp8": 
            self.accel.save_model(self.model, save_dir)
            return 

        self.accel.wait_for_everyone() 

        self.accel.unwrap_model(self.model).to(torch.bfloat16).save_pretrained(
            save_dir, 
            is_main_process=self.accel.is_main_process,
            save_function=self.accel.save,
        )

        if isinstance(self.accel.unwrap_model(self.model), PeftModel): 
            self.accel.unwrap_model(self.model).peft_config["default"].save_pretrained(save_dir)


    def load_best_checkpoint(self) -> PreTrainedModel: 
        """
        load_from_model_object: In case the PreTrainedModel class has some
            post-initialization modifications, directly load state_dict into 
            load_from_model_object. Otherwise, self.model.__class__.from_pretrained is used.
        """
        self.model = self.accel.unwrap_model(self.model)
        self.model.zero_grad(set_to_none=True)
        if len(self.model.active_adapters()) > 0: 
            self.model.load_adapter(os.path.join(self.output_dir, "best_tfmr"), adapter_name="best")
            self.model.set_adapter("best")
        else: 
            self.model = self.model.from_pretrained(os.path.join(self.output_dir, "best_tfmr"))
        self.model, dummy_optim = self.accel.prepare(self.model, torch.optim.AdamW(self.model.parameters(), lr=1e-5))

        return self.model





 

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
    
        
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
    





class MICompressTrainer(SimpleTrainer): 
    def __init__(self, 
                 *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.alpha = self.args.alpha 

     
    def __post_init__(self): 
        print("========= MI COMPRESS TRAINING =========")
    
    def compute_loss(self, batch, return_loss_breakdown=True):
        
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"], 
            attention_mask=batch["attention_mask"],
            paraphrase_input_ids=batch["paraphrase_input_ids"], 
            paraphrase_attention_mask=batch["paraphrase_attention_mask"], 
            use_cache=False)

        # print("input_ids shape:", batch["input_ids"].shape) 
        # print("paraphrase_input_ids shape:", batch["paraphrase_input_ids"].shape) 
        ce_loss = outputs.loss
        pos, neg = outputs.critic_outputs

        del outputs

        dv_loss = -(pos.mean() - torch.log(neg.exp().mean() + 1e-9))

        if return_loss_breakdown: 
            mets = {"loss": (ce_loss + self.alpha * dv_loss).item(), 
                    "ce_loss": ce_loss.item(), 
                    "mi_lb": -dv_loss.item()} 
            return ce_loss + self.alpha * torch.minimum(dv_loss, torch.tensor(1, device=dv_loss.device)), mets
        return ce_loss + self.alpha * torch.minimum(dv_loss, torch.tensor(1, device=dv_loss.device))



@dataclass
class StateCache: 
    input_hidden_states: Tuple[torch.FloatTensor]
    input_attention_mask: torch.Tensor
    paraphrase_hidden_states: Tuple[torch.FloatTensor] 
    paraphrase_attention_mask: torch.Tensor
    first_gist_idx: torch.Tensor
    last_gist_idx: torch.Tensor 

    def to(self, device): 
        self.input_hidden_states = [x.to(device) for x in self.input_hidden_states]
        self.input_attention_mask = self.input_attention_mask.to(device)
        self.paraphrase_hidden_states = [x.to(device) for x in self.paraphrase_hidden_states]
        self.paraphrase_attention_mask = self.paraphrase_attention_mask.to(device)
        self.first_gist_idx = self.first_gist_idx.to(device)
        self.last_gist_idx = self.last_gist_idx.to(device)
        return self
    def detach(self): 
        self.input_hidden_states = [x.detach() for x in self.input_hidden_states]
        self.input_attention_mask = self.input_attention_mask.detach()
        self.paraphrase_hidden_states = [x.detach() for x in self.paraphrase_hidden_states]
        self.paraphrase_attention_mask = self.paraphrase_attention_mask.detach()
        self.first_gist_idx = self.first_gist_idx.detach()
        self.last_gist_idx = self.last_gist_idx.detach()
        return self




class MICompressTrainerWithCoordinateDescent(SimpleTrainer): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

        if not hasattr(self.args, "compress_fraction"): 
            self.args.compress_fraction = 0.25 

        self.compress_fraction = self.args.compress_fraction
        self.hidden_state_cache = []

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.c_accel = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps, 
            step_scheduler_with_optimizer=False, 
            kwargs_handlers=[ddp_kwargs]
        )
        self.model = self.accel.unwrap_model(self.model)
        self.gist_token_id = self.model.gist_token_id

        critic = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 1, bias=False)
        ).to(self.c_accel.device).to(torch.bfloat16) #TODO: make more general

        self.model.add_critic_network(critic)
        
        params = []
        names = []
        for n, p in self.model.named_parameters(): 
            if p.requires_grad:
                params.append(p)
                names.append(n)

        self.c_optim = torch.optim.AdamW(params, lr=self.args.lr) 
        self.model, self.c_optim = self.c_accel.prepare(self.model, self.c_optim)




    def __post_init__(self):
        print("========= MI Compress Trainer with Coordinate Descent =========")


    def sample_dataloader(self, 
                            iter_dataloader: Iterable, 
                            dataloader: DataLoader) -> Tuple[BatchEncoding, Iterable]: 
        """
        Samples from an iterable dataloader. If StopIteration is reached, 
        then rebuild the dataloader and continue to sample. 
        """
        try: 
            return next(iter_dataloader), iter_dataloader
        except StopIteration:
            iter_dataloader = iter(dataloader)
            return next(iter_dataloader), iter_dataloader

    def get_base_model(self, model: PreTrainedModel, accel: Accelerator): 
        model = accel.unwrap_model(model)
        model.disable_adapters() 
        model = accel.prepare_model(model) 
        return model 
    
    def get_peft_model(self, model: PreTrainedModel, accel: Accelerator): 
        model = accel.unwrap_model(model) 
        model.enable_adapters() 
        model = accel.prepare_model(model) 
        return model

    def train(self): 

        self.model.train() 
        
        iter_train_dloader = iter(self.train_dloader) 
        n_total_steps = self.epochs * len(self.train_dloader) 
        n_compress_steps = int(self.compress_fraction * n_total_steps) 
        n_task_steps = n_total_steps - n_compress_steps

        compress_loop = tqdm.tqdm(range(n_compress_steps), desc=f"Compressing") 

        ref_model = self.get_base_model(self.model, self.c_accel)
        ref_model.eval()

        if self.accel.unwrap_model(self.model) == self.model: 
            get_pos_critic_outputs = self.model.get_pos_critic_outputs
            get_neg_critic_outputs = self.model.get_neg_critic_outputs
        else: 
            get_pos_critic_outputs = self.model.module.get_pos_critic_outputs
            get_neg_critic_outputs = self.model.module.get_neg_critic_outputs


        for i in compress_loop: 
            self.model.train()

            batch, iter_train_dloader = self.sample_dataloader(iter_dataloader=iter_train_dloader, dataloader=self.train_dloader)
            batch = self.preprocess_fn(batch, self.tokenizer, use_chat_template=False).to(self.c_accel.device)

            with torch.no_grad():
                # self.model = self.get_base_model(self.model, self.c_accel)
                state_cache = self.collect_hidden(batch, model=ref_model).detach()
                # self.model = self.get_peft_model(self.model, self.c_accel) 

            with self.accel.accumulate(self.model):
                with self.accel.autocast(): 
                    pos, gist_states = get_pos_critic_outputs(
                        input_hidden=state_cache.input_hidden_states, 
                        input_attention_mask=state_cache.input_attention_mask,
                        first_gist_index = state_cache.first_gist_idx, 
                        last_gist_index = state_cache.last_gist_idx, 
                        return_gist_states=True)
                    neg = get_neg_critic_outputs( 
                        paraphrase_hidden=state_cache.paraphrase_hidden_states, 
                        paraphrase_attention_mask=state_cache.paraphrase_attention_mask, 
                        input_gist_states=gist_states
                    )

                    dv_loss = -(pos.mean() - torch.log(neg.exp().mean() + 1e-9))
                    loss = torch.maximum(dv_loss, -torch.tensor(25, device=pos.device)) 
                
                self.c_accel.backward(loss)
                loss.detach()
                self.c_optim.step() 
                self.c_optim.zero_grad()
                

            del state_cache
            compress_loop.set_postfix({"dv_lower_bound": -dv_loss.item()})


        task_loop = tqdm.tqdm(range(n_task_steps), desc=f"Training {0.0}/{self.epochs} Epochs")
        self.model = self.c_accel.unwrap_model(self.model)
        # self.optim = self.c_optim.optimizer 
        self.model = self.accel.prepare_model(self.model)

        for i in task_loop: 

            batch, iter_train_dloader = self.sample_dataloader(iter_dataloader=iter_train_dloader, dataloader=self.train_dloader)
            batch = self.preprocess_fn(batch, self.tokenizer, use_chat_template=False).to(self.c_accel.device)

            with self.accel.accumulate(self.model):
                with self.accel.autocast():

                    loss, loss_mets = self.compute_loss(batch, return_loss_breakdown=True)
                    # loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), batch["labels"].view(-1))

                self.accel.backward(loss)
                loss = loss.detach() 

                # loss.backward()
                self.optim.step() 
                self.lr_scheduler.step()
                self.optim.zero_grad()
                # torch.cuda.empty_cache()

            self.accel.wait_for_everyone()

            # update tbar desc
            cur_epoch = i / n_task_steps * self.epochs 
            task_loop.set_description(f"Training {cur_epoch:.2f}/{self.epochs} Epochs")
            # update tbar postfix
            if len(loss_mets) > 0:
                loss_mets_stringify = {k: f"{v:.4f}" for k, v in loss_mets.items()} 
                task_loop.set_postfix(loss_mets_stringify | {"lr": f"{self.lr_scheduler.get_last_lr()[0]:.4e}"})
            else: 
                task_loop.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.lr_scheduler.get_last_lr()[0]:.4e}"})



            if i%self.eval_steps == 0 and i > 0: 
                eval_preds = self.evaluate(generate=True)
                self.accel.wait_for_everyone()
                # DEBUG 
                os.makedirs(os.path.join(self.output_dir, "preds"), exist_ok=True)
                if self.accel.is_main_process: 
                    print(eval_preds.predictions[:2])
                with open(os.path.join(self.output_dir, "preds", f"{self.accel.process_index}.log"), "w") as f: 
                    [f.write(repr(f"{self.tokenizer.decode(torch.where(p == -100, self.tokenizer.pad_token_id, p), skip_special_tokens=True)}") + "\n") for p in eval_preds.predictions]

                os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)
                if self.accel.is_main_process: 
                    print(eval_preds.label_ids[:2])
                with open(os.path.join(self.output_dir, "labels", f"{self.accel.process_index}.log"), "w") as f: 
                    [f.write(repr(f"{self.tokenizer.decode(torch.where(p == -100, self.tokenizer.pad_token_id, p), skip_special_tokens=True)}") + "\n") for p in eval_preds.label_ids]


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
                    self.accel.wait_for_everyone()
                    self.save_checkpoint(save_dir = os.path.join(self.output_dir, "best_tfmr"))
                    self.accel.wait_for_everyone()
                    if self.accel.is_main_process: 
                        with open(os.path.join(self.output_dir, "best_metrics.log"), "a") as f: 
                            f.write(f"step:{i},")
                            f.write(",".join([f"{k}:{v}" for k, v in metrics.items()]))
                            f.write("\n")
                


    def collect_hidden(self, 
                       batch: BatchEncoding, 
                       model: PreTrainedModel): # TODO: once MI Model has its own class, use it here.
        with torch.no_grad(): 
            with self.accel.autocast(): 
                outputs = self.model(input_ids=batch["input_ids"], 
                            attention_mask=batch["attention_mask"]) 
                input_hidden_states = outputs.hidden_states

                p_outputs = self.model(input_ids=batch["paraphrase_input_ids"], 
                                       attention_mask=batch["paraphrase_attention_mask"])
                paraphrase_hidden_states = p_outputs.hidden_states 

                first_gist_idx = get_first_idx_of_token(batch["input_ids"], self.gist_token_id)
                last_gist_idx = get_last_idx_of_token(batch["input_ids"], self.gist_token_id)

                state_cache = StateCache(
                    input_hidden_states, 
                    batch["attention_mask"], 
                    paraphrase_hidden_states, 
                    batch["paraphrase_attention_mask"], 
                    first_gist_idx, last_gist_idx
                )

                # self.hidden_state_cache.append(state_cache)
        return state_cache
                


    def compute_loss(self, 
                        batch: BatchEncoding, 
                        return_loss_breakdown: Optional[bool] =True, 
                        **custom_kwargs) -> Tuple[torch.FloatTensor, Optional[Dict[str, float]]]: 
        


        assert "paraphrase_input_ids" in batch, f"critic loss requires paraphrase_input_ids" 
        outputs = self.model(input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"], 
            paraphrase_input_ids=batch["paraphrase_input_ids"], 
            paraphrase_attention_mask=batch["paraphrase_attention_mask"])

        loss = outputs.loss 
        mets = {"ce_loss": loss.item()}

        self.accel.wait_for_everyone()
        
        if return_loss_breakdown: 
            return loss, mets 
        return loss, None

    
    def rewrap_optim_model(self): 
        self.model = self.accel.unwrap_model(self.model)
        self.optim = self.optim.optimizer

        self.model, self.optim = self.accel.prepare(self.model, self.optim) 
        return self.model, self.optim


    def train_critic(self): 
        num_steps = len(self.hidden_state_cache)
        tbar = tqdm.tqdm(range(num_steps), desc=f"Training Critic {0}/{num_steps}")

        self.rewrap_optim_model()

        if self.accel.unwrap_model(self.model) == self.model: 
            get_pos_critic_outputs = self.model.get_pos_critic_outputs
            get_neg_critic_outputs = self.model.get_neg_critic_outputs
        else: 
            get_pos_critic_outputs = self.model.module.get_pos_critic_outputs
            get_neg_critic_outputs = self.model.module.get_neg_critic_outputs


        for i in tbar: 
            self.model.train() 
            state_cache: StateCache = self.hidden_state_cache[i].to(self.accel.device)
            with self.accel.accumulate(self.model):
                with self.accel.autocast(): 
                    pos, gist_states = get_pos_critic_outputs(
                        input_hidden=state_cache.input_hidden_states, 
                        input_attention_mask=state_cache.input_attention_mask,
                        first_gist_index = state_cache.first_gist_idx, 
                        last_gist_index = state_cache.last_gist_idx, 
                        return_gist_states=True)
                    neg = get_neg_critic_outputs( 
                        paraphrase_hidden=state_cache.paraphrase_hidden_states, 
                        paraphrase_attention_mask=state_cache.paraphrase_attention_mask, 
                        input_gist_states=gist_states
                    )

                    dv_loss = -(pos.mean() - torch.log(neg.exp().mean() + 1e-9))
                    loss = torch.maximum(dv_loss, -torch.tensor(1, device=pos.device))
                self.accel.backward(loss)
                dv_loss = dv_loss.detach() 

                # for name, p in self.model.named_parameters():
                #     if p.grad is not None and "critic" in name:
                #         print(name, p.dtype, p.grad.dtype)

                # if i > 0 and i+1 % 10 == 0:
                self.optim.step() 
                self.lr_scheduler.step() 
                self.optim.zero_grad() 

            self.accel.wait_for_everyone() 

            del state_cache 
            self.hidden_state_cache[i] = None 
            torch.cuda.empty_cache()

            tbar.set_description(f"Training Critic {i+1}/{num_steps}")
            tbar.set_postfix({"dv_lower_bound": -dv_loss.item()})
        self.hidden_state_cache = []

        self.rewrap_optim_model()


        

if __name__ == "__main__": 

    from .tasks import WikitextTask 

    wikitext = WikitextTask()
    train_dloader = wikitext.get_dataloaders(["train"])[0]
    batch  = next(iter(train_dloader))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") 
    
    output = DEFAULT_CAUSAL_LM_PREPROCESS_FN(batch, tok) 

