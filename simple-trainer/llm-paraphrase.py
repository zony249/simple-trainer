import os 
import sys 
from typing import Dict, List, Tuple, Optional, Any
from argparse import ArgumentParser, Namespace
import json
import tqdm

import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 
from .tasks import TASK_MAP 
from .tasks.alpaca_plus.utils import DataCollatorForAlpacaCLM

if __name__ == "__main__": 

    parser = ArgumentParser(description="Paraphrase Generator")
    parser.add_argument("--hf_name_or_path", type=str, required=True, help="Huggingface model used to generate paraphrase")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--task", type=str, choices = ["alpaca_plus"])
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_name_or_path, 
        torch_dtype=torch.bfloat16, 
        device_map = "auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side="left"

    if args.task == "alpaca_plus":
        splits = ["train", "validation_seen", "validation_unseen", "validation_human"]

        list_dataloaders = TASK_MAP["alpaca_plus"](
                                    splits = splits
                                ).get_dataloaders(splits)
        dict_dataloaders = {k: v for k, v in zip(splits, list_dataloaders)} 

        train_dloader = dict_dataloaders["train"] 
        # train_dloader = torch.utils.data.DataLoader(train_dloader.dataset,
        #                         batch_size=train_dloader.batch_size,shuffle=False)         

        dataset_splits = TASK_MAP["alpaca_plus"](splits=splits).get_splits(splits) 
        train_dloader = torch.utils.data.DataLoader(dataset_splits["train"],
                                batch_size=128,
                                shuffle=False, 
                                collate_fn=lambda x: x)    

        tokenizer.add_special_tokens({"additional_special_tokens":["<GIST>"]})
        data_collate_fn = DataCollatorForAlpacaCLM(
            tokenizer=tokenizer, 
            max_length=1024, 
            max_length_human=1536, 
            label_pad_token_id=-100,
            gist_token = len(tokenizer)-1, 
            pad_token=tokenizer.pad_token_id, 
            add_gist_token = True, 
            num_gist_tokens=1, 
            check_correctness = False, 
        )
    else: 
        raise NotImplementedError 

    def insert_chat_template_prompt(tokenizer, prompt, batch_tensors): 
        assert isinstance(batch_tensors, torch.Tensor), f"batch_tensors must be of type torch.Tensor"
        assert batch_tensors.ndim > 1, "number of dimensions of batch_tensors must be greater than 1"

        batch_seqs = tokenizer.batch_decode(batch_tensors, skip_special_tokens=False) 
        batch_prompted = []
        for seq in batch_seqs: 
            before_gist_idx = seq.find("<GIST>")
            after_gist_idx = seq.rfind("<GIST>") + len("<GIST>")
            seq_before_gist = seq[:before_gist_idx]
            seq_after_gist = seq[after_gist_idx:]

            seq_before_gist = tokenizer.decode(tokenizer.encode(seq_before_gist), skip_special_tokens=True)

            chat = [
                {"role": "system", "content": "A conversation between a user and an assistant. As the assistant, you will help answer the user's queries. Be precise."}, 
                {"role": "user", "content": prompt + "\n" + seq_before_gist}, 
            ]
            res = tokenizer.apply_chat_template(chat, 
                                                tokenize=False, 
                                                add_generation_prompt=True, 
                                                enable_thinking=False)
            batch_prompted.append(res)

        batch_prompted  

        return tokenizer(batch_prompted, 
                         padding="longest", 
                         return_tensors="pt") 

    #TODO: in the end, write out to json file
    os.makedirs(args.save_dir, exist_ok=True)
    save_file = os.path.join(args.save_dir, "paraphrases") 
    if os.path.exists(save_file): 
        os.remove(save_file)

    os.makedirs(args.save_dir, exist_ok=True) 
    json_file = os.path.join(args.save_dir, "alpaca_pp.json") 
    if os.path.exists(json_file): 
        os.remove(json_file)
    json_container = []
    

    pbar = tqdm.tqdm(train_dloader, desc="Paraphrasing")
    prompt = "Paraphrase the following text, including the prompt instruction prefix (i.e., \"Instruction:\"), but make sure you don't include the prefix \"Paraphrase:\" in the result. Make sure that the paraphrase does not contain additional unnecessary information:"


    for batch in pbar: 

        collated_batch = data_collate_fn(batch) 
        
        prompted_batch = insert_chat_template_prompt(tokenizer, 
                                        prompt=prompt, 
                                        batch_tensors=collated_batch["prompt_input_ids"]).to("cuda")
        
        outputs = model.generate(**prompted_batch, max_new_tokens=1024, do_sample=False, num_beams=1)

        think_id = tokenizer.convert_tokens_to_ids("</think>")

        for i, line in enumerate(outputs): 
            location = (line ==think_id).nonzero(as_tuple=True)[0].item()
            paraphrase = tokenizer.decode(line[location+1:], skip_special_tokens=True).strip()

            # file output            
            with open(save_file, "a") as f:
                f.write(paraphrase) 
                f.write("\n")

            # json output with original texts
            batch[i]["paraphrase"] = paraphrase 
            json_container.append(batch[i])

        # break

    json_string = json.dumps(json_container)
    with open(json_file, "w") as f: 
        f.write(json_string) 
    