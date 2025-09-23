import os 
import sys
import dataclasses 
from copy import deepcopy 
from typing import Dict, List, Tuple, Union, Self, Optional, Any
import json

import torch 
from torch.utils.data import Dataset, DataLoader

# sys.path.append("simple-trainer")
from ..abstract_task import AbstractTask 


import datasets
from datasets import load_dataset, load_from_disk
from datasets.splits import NamedSplit

from .utils import AlpacaPlusOrig, DataCollatorForAlpacaCLM

logger = datasets.logging.get_logger(__name__)



class AlpacaPlus(AbstractTask): 
    task = "alpaca_plus"
    dataset_path = ""
    dataset_name = ""

    def __init__(self, 
                 splits: List[str] = ["train", "validation_seen", "validation_unseen", "validation_human"], 
                 load_from_disk=False, 
                 local_dir=None, 
                 **kwargs): 


        self.builder = AlpacaPlusOrig()
        self.builder.download_and_prepare()

        # gets_splits and process_dataset called in parent init
        super().__init__(splits=splits, 
                         load_from_disk=load_from_disk, 
                         local_dir=local_dir, 
                         **kwargs) 
    def get_dataset(self, 
                    **kwargs) -> Dataset: 
        dataset = []
        if self.load_from_disk: 
            pass
        else: 
            pass
        return dataset

    def get_splits(self, 
                   list_splits: List[str], 
                   **kwargs) -> Dict[str, Dataset]: 
        dict_datasets = {}
        for split in list_splits: 
            if split not in ["train", "validation_seen", "validation_unseen", "validation_human"]:
                raise ValueError(f"Split {split} not recognized. Available splits are 'train', 'validation_seen', 'validation_unseen', 'validation_human'")
            ds = self.builder.as_dataset(split=split)  # to make sure the split is generated 
            dict_datasets[split] = ds 
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
        Does nothing. Used by the dataloader. Since this dataset is used for instruction tuning, 
        which involves specific data preprocessing before collation, we carry this out in DataCollatorForAlpacaCLM.
        """ 
        return batch
        
    def process_dataset(self, 
                        dataset:Dataset) -> Dataset: 
        return dataset

    def compute_metrics(self, example):
        return super().compute_metrics(example)










