import os 
import sys 
import torch 
from transformers import ( 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModel,  
)
from .modeling_cllama import CLlamaForCausalLM, CLlamaModel
from .configuration_cllama import CLlamaConfig

if __name__ == "__main__":

    # config = CLlamaConfig.from_pretrained("huggyllama/llama-7b")
    AutoConfig.register("cllama", CLlamaConfig)
    AutoModel.register(CLlamaConfig, CLlamaModel)
    AutoModelForCausalLM.register(CLlamaConfig, CLlamaForCausalLM)
    
    model = CLlamaForCausalLM.from_pretrained("huggyllama/llama-7b")
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
