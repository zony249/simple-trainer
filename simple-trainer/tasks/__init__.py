# from .abstract_task import AbstractTask
from .wikitext import WikitextTask
from .alpaca_plus import AlpacaPlus
from .alpaca_plus.utils import DataCollatorForAlpacaCLM

TASK_MAP = {
    "wikitext": WikitextTask,
    "alpaca_plus": AlpacaPlus,
}