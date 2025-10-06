import os 
import sys 
from typing import Dict, Any, Optional, Self, List, Tuple, Union 

import torch 
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer
)

from .models import (
    CLlamaConfig,
    CLlamaModel,  
    CLlamaForCausalLM
)

COMPRESSED_MODEL_MAP = {
    "qwen3": "cqwen3", 
    "llama" : "cllama"
}


# import debugpy 
# debugpy.listen(("172.26.93.93", 5678))
# debugpy.wait_for_client()


def get_compression_model(model: PreTrainedModel, 
                          tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: 
    config = model.config 
    model_type = config.model_type 
    if model_type not in COMPRESSED_MODEL_MAP: 
        raise ValueError(f"Model type {model_type} is not supported for compression.") 
    compression_model_type = COMPRESSED_MODEL_MAP[model_type] 

    if compression_model_type == "cllama": 
        cmodel_config = CLlamaConfig.from_pretrained(model.config._name_or_path)
        cmodel_config.torch_dtype = torch.bfloat16
        cmodel = CLlamaForCausalLM(cmodel_config)
        cmodel.load_state_dict(model.state_dict(), strict=False)
        
        ctokenizer = cmodel.enable_compression_mode(tokenizer=tokenizer) 
        
        # del model
        # torch.cuda.empty_cache()
        
    return cmodel, ctokenizer

def get_mi_compression_model(model: PreTrainedModel, 
                             tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedModel, PreTrainedTokenizer]: 
    pass

if __name__ == "__main__": 

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import infer_auto_device_map, dispatch_model
    model_name = "llama-7b"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    compression_model, compression_tokenizer = get_compression_model(model, tokenizer)


    device_map = infer_auto_device_map(compression_model, 
                                        max_memory={0: "20GB", 1: "20GB"}, 
                                        no_split_module_classes=compression_model._no_split_modules)
    dispatch_model(compression_model, device_map=device_map)
    # compression_model = compression_model

    text = ["Hello, my dog is cute <GIST><GIST> his name is joe.", "Hey everyone! what's up? <GIST>"]
    device = next(iter(compression_model.parameters())).device.type
    inputs = compression_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = compression_model.generate(**inputs, use_cache=True, max_new_tokens=100)
    print(compression_tokenizer.batch_decode(outputs))

    # print(compression_model)
    # print(compression_tokenizer)