from . import alpaca_plus
from .alpaca_plus import AlpacaPlus
from .utils import AlpacaPlusOrig


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

    task = AlpacaPlus() 
    print(task.splits)

    pass 