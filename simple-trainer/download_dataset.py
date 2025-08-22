# Download dataset to local machine. Might be useful for compute nodes that
# do not have internet access.
import os 
from argparse import ArgumentParser, Namespace 

from datasets import load_dataset, load_from_disk

if __name__ == "__main__": 
    parser = ArgumentParser("Download dataset to local machine.")
    parser.add_argument("name", type=str)

    args = parser.parse_args() 

    save_dir = args.name + "_local"

    if args.name == "wikitext": 
        train = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="train")
        val = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="validation")
        test = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test")

        train.save_to_disk(os.path.join(save_dir, "train"))
        val.save_to_disk(os.path.join(save_dir, "validation"))
        test.save_to_disk(os.path.join(save_dir, "test"))
