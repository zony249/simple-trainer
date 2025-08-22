import os 
import dataclasses 
from copy import deepcopy
from typing import Dict, List, Union, Optional, Self, Tuple
from abc import abstractmethod, ABCMeta

import torch 
from torch.utils.data import Dataset 

class AbstractTask(metaclass=ABCMeta): 
    task = None
    dataset_path = None
    dataset_name = None
    def __init__(self, 
                 splits: List[str], 
                 load_from_disk=False, 
                 **kwargs): 
        self.load_from_disk = load_from_disk 
        self.splits = self.get_splits(splits, **kwargs)

        for k in self.splits: 
            self.splits[k] = self.process_dataset(self.splits[k])

    @abstractmethod
    def get_dataset(self, 
                    **kwargs) -> Dataset: 
        raise NotImplementedError

    @abstractmethod 
    def get_splits(self, 
                   list_splits: List[str], 
                   **kwargs) -> Dict[str, Dataset]: 
        raise NotImplementedError
    
    @abstractmethod 
    def process_dataset(self, 
                        dataset:Dataset) -> Dataset: 
        raise NotImplementedError