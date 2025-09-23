import os 
import sys 
from typing import Dict, Any, Optional, Self, List, Tuple, Union 

import torch 
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer
)

COMPRESSED_MODEL_MAP = {
    "qwen3": "cqwen3"
}

def get_compression_model(model: PreTrainedModel, 
                          tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: 
    pass

def get_mi_compression_model(model: PreTrainedModel, 
                             tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: 
    pass