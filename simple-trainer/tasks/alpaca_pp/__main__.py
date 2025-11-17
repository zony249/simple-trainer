from . import alpaca_pp
from .alpaca_pp import AlpacaPlusPlus
from .utils import AlpacaPlusPlusOrig, DataCollatorForAlpacaPlusPlusCLM


if __name__ == "__main__": 

    # ALL THIS IS FOR TESTING PURPOSES 
    # builder = AlpacaPlusOrig()
    # builder.download_and_prepare()
    # print(builder.info)
    # train = builder.as_dataset(split="train")
    # val_seen = builder.as_dataset(split="validation_seen")
    # val_unseen = builder.as_dataset(split="validation_unseen")
    # val_human = builder.as_dataset(split="validation_human")




    # print(ds)
    # pass

    from transformers import AutoTokenizer 
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tok.add_special_tokens({"additional_special_tokens": ["<GIST>"]})

    task = AlpacaPlusPlus()
    print(task.splits)

    preprocess_fn = DataCollatorForAlpacaPlusPlusCLM(tok, 
                                                    max_length=1000, 
                                                    max_length_human=1000, 
                                                    label_pad_token_id=-100, 
                                                    return_tensors="pt", 
                                                    gist_token=tok.convert_tokens_to_ids("<GIST>"), 
                                                    pad_token=tok.pad_token_id,
                                                    add_gist_token=True, 
                                                    num_gist_tokens=1, 
                                                    check_correctness=False)

     
    list_splits = ["train", "validation_unseen"]
    list_dataloaders = task.get_dataloaders(list_splits)
    dataloaders = {k:v for k, v in zip(list_splits, list_dataloaders)} 

    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["validation_unseen"]
    
    train_batch = next(iter(train_dataloader))
    batched = preprocess_fn(train_batch) 

    val_batch = next(iter(val_dataloader))
    val_batched = preprocess_fn(val_batch)

    pass 