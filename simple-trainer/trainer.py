

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel, 
)


class MyTrainer: 
    def __init__(self, 
                 model: PreTrainedModel, 
                 optim : Optimizer, 
                 lr_sched : LRScheduler, 
                 trainset : Dataset, 
                 valset : Dataset): 
        pass 