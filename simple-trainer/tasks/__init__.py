# from .abstract_task import AbstractTask
from .wikitext import WikitextTask
from .alpaca_plus import AlpacaPlus
from .alpaca_plus.utils import DataCollatorForAlpacaCLM
from .alpaca_pp import AlpacaPlusPlus
from .alpaca_pp.utils import DataCollatorForAlpacaPlusPlusCLM

TASK_MAP = {
    "wikitext": WikitextTask,
    "alpaca_plus": AlpacaPlus,
    "alpaca_pp": AlpacaPlusPlus
}

COMPRESSION_TASKS = [
    "alpaca_plus", 
    "alpaca_pp"
]