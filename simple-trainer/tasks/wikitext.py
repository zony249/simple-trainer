import os 
import dataclasses 
from copy import deepcopy 
from typing import Dict, List, Tuple, Union, Self, Optional, Any

import torch 
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset, load_from_disk

from .abstract_task import AbstractTask 

class WikitextTask(AbstractTask): 
    task = "wikitext"
    dataset_path = "EleutherAI/wikitext_document_level"
    dataset_name = "wikitext-2-raw-v1"

    def __init__(self, 
                 splits: List[str] = ["train", "test", "validation"], 
                 load_from_disk=False, 
                 local_dir=None, 
                 **kwargs): 
        super().__init__(splits=splits, 
                         load_from_disk=load_from_disk, 
                         local_dir=local_dir, 
                         **kwargs) 
    def get_dataset(self, 
                    **kwargs) -> Dataset: 
        if self.load_from_disk: 
            assert self.local_dir is not None, f"local_dir must be defined if load_from_disk is True."
            dataset = load_from_disk(os.path.join(self.local_dir, kwargs["split"]))
        else: 
            dataset = load_dataset(path=self.dataset_path, 
                                   name=self.dataset_name, 
                                   **kwargs)
        return dataset
    def get_splits(self, 
                   list_splits: List[str], 
                   **kwargs) -> Dict[str, Dataset]: 
        dict_datasets = {}
        self.splits = getattr(self, "splits", None)
        if self.splits is not None:
            return {k: v for k, v in self.splits.items() if k in list_splits}
        for split in list_splits: 
            kwargs["split"] = split 
            dataset = self.get_dataset(**kwargs)
            dict_datasets[split] = dataset
        return dict_datasets

    def get_dataloaders(self, 
                        list_splits: List[str], 
                        batch_size=8, 
                        **kwargs) -> List[DataLoader]: 
        data_splits = self.get_splits(list_splits, **kwargs)
        dataloaders = []
        for split in list_splits: 
            if split == "train":
                dataloaders.append(DataLoader(
                    dataset=data_splits[split], 
                    batch_size=batch_size,
                    shuffle=True, 
                    collate_fn=self.collate_fn
                ))
            else: 
                dataloaders.append(DataLoader(
                    dataset=data_splits[split], 
                    batch_size=batch_size,
                    collate_fn=self.collate_fn
                ))
        return dataloaders 

    def collate_fn(self, batch: Tuple) -> Any:
        """
        simply unwrap the docs
        """ 
        batch 
        batch = [b["page"] for b in batch] 
        return batch
        
    def process_dataset(self, 
                        dataset:Dataset) -> Dataset: 
        new_dataset = dataset.map(self.preprocess_fn) 
        return new_dataset 

    def preprocess_fn(self, example): 
        example["page"] = wikitext_detokenizer(example)
        return example    

    def compute_metrics(self, example):
        return super().compute_metrics(example)


# From LM-eval


import re


def wikitext_detokenizer(doc):
    string = doc["page"]
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def process_results(doc, results):
    (loglikelihood,) = results
    # IMPORTANT: wikitext counts number of words in *original doc before detokenization*
    _words = len(re.split(r"\s+", doc["page"]))
    _bytes = len(doc["page"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }

if __name__ == "__main__": 
    wikitext = WikitextTask(load_from_disk=True, local_dir="wikitext_local")
    print(wikitext.get_splits(["train", "validation"]))
    train_dataloader = wikitext.get_dataloaders(["train"])["train"] 

    sample = next(iter(train_dataloader))
    pass