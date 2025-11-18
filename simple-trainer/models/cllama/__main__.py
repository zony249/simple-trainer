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
    tokenizer.padding_side = "left"

    # turn on gist mode
    tokenizer = model.enable_compression_model(tokenizer=tokenizer,
                                gist_masking=True)

    test_sents = ["first sentence.<GIST>\n\none gist token.", 
     "second sentence. gist tokens: <GIST><GIST>\n\ntwo gists.", 
     "third sent. no gist tokens"]

    inputs = tokenizer(test_sents, return_tensors="pt")

    model(**inputs)
    pass