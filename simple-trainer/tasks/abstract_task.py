import os 
import dataclasses 
from copy import deepcopy
from typing import Dict, List, Union, Optional, Self, Tuple, Any
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
                 local_dir=None, 
                 **kwargs): 
        """
        An abstract class for tasks. All subclasses must: 
            1. be loadable from disk OR online (or both)
            2. allow loading different splits 
            3. accept pre-processing (if not, make process_dataset an identity function)
        """
        self.load_from_disk = load_from_disk 
        self.local_dir = local_dir
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
    
    @abstractmethod 
    def compute_metrics(self, example) -> Any: 
        raise NotImplementedError