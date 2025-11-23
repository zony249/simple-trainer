import os 
import sys 
import torch 
from transformers import ( 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModel,  
)
from .modeling_cllama import CLlamaForCausalLM, CLlamaModel, MICLlamaForCausalLM
from .configuration_cllama import CLlamaConfig

if __name__ == "__main__":

    if int(os.environ["DEBUGPY_ENABLE"]) == 1:
        import debugpy 
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            debugpy.listen(("172.26.93.164", 5678 + local_rank))
            debugpy.wait_for_client()

    # config = CLlamaConfig.from_pretrained("huggyllama/llama-7b")
    AutoConfig.register("cllama", CLlamaConfig)
    AutoModel.register(CLlamaConfig, CLlamaModel)
    AutoModelForCausalLM.register(CLlamaConfig, CLlamaForCausalLM)
    
    model = MICLlamaForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # turn on gist mode
    tokenizer = model.enable_compression_mode(tokenizer=tokenizer,
                                gist_masking=True)

    input_sents = ["first sentence.<GIST><GIST>\n\none gist token.", 
     "second sentence. gist tokens: <GIST><GIST>\n\ntwo gists.", 
     "third sent. <GIST><GIST>\n\nno gist tokens"]
    paraphrase_sents = ["first sentence.\n\none gist token.", 
     "second sentence. gist tokens: \n\ntwo gists.", 
     "third sent. no gist tokens"]

    inputs = tokenizer(input_sents, return_tensors="pt", padding=True, truncation=True, )
    para = tokenizer(paraphrase_sents, return_tensors="pt", padding=True, truncation=True, )

    outputs = model(input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    paraphrase_input_ids=para["input_ids"], 
                    paraphrase_attention_mask=para["attention_mask"])
    pass
    