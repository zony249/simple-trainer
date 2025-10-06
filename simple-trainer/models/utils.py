import os 
import sys 
from typing import Dict, List, Union, Any, Self, Optional, Tuple 


import torch 
from torch import nn 
import torch.nn.functional as F 
from transformers import Cache

def get_first_idx_of_token(input_ids:torch.Tensor, 
                           token_id: int) -> torch.Tensor: 
    """
    output: torch.Tensor[batch_size]: an integer for each sequence denoting 
        index of the first occurance of token_id
    """
    equals = input_ids == token_id
    return torch.argmax((input_ids == token_id).int(), dim=-1) 

def get_last_idx_of_token(input_ids:torch.Tensor, 
                          token_id: int) -> torch.Tensor: 
    """
    output: torch.Tensor[batch_size]: an integer for each sequence denoting 
        index of the last occurance of token_id
    """
    _, seq_len = input_ids.shape
    return seq_len - torch.argmax(torch.flip((input_ids == token_id).int(), dims=(-1,)), dim=-1) - 1


def get_causal_mask(attention_mask:torch.Tensor, 
                    num_attn_heads:int, 
                    total_seq_len:int) -> torch.Tensor: 
    pre_tril = attention_mask[:, None, None, :] \
            * torch.ones((1, num_attn_heads, total_seq_len, total_seq_len)).to(attention_mask.device) 
    causal_mask = torch.tril(pre_tril).bool()
    return causal_mask


def get_causal_mask_with_cache(cache: Cache, 
                               attention_mask: torch.Tensor, 
                               num_attn_heads: int, 
                               num_new_toks: int, 
                               total_seq_len: int
                               ) -> torch.Tensor: 
    cache_seq_len = cache.get_seq_length() 
    pre_tril = attention_mask[:, None, None, :] * torch.ones((1, num_attn_heads, num_new_toks, total_seq_len)).to(attention_mask.device)
    causal_mask = torch.tril(pre_tril, diagonal=cache_seq_len).bool()
    return causal_mask

if __name__ == "__main__": 
    import debugpy 
    debugpy.listen(("172.26.93.100", 5678)) 
    debugpy.wait_for_client() 

    x = torch.tensor([[1, 2, 3, 4, 5, 3, 5, 7], 
                      [1, 2, 5, 5, 3, 5, 6, 7]])


    token_id = 3
    
    first = get_first_idx_of_token(x, token_id)
    last = get_last_idx_of_token(x, token_id)

    pass
