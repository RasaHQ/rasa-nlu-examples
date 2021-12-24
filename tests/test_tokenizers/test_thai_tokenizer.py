import pytest

import itertools as it

from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa_nlu_examples.tokenizers import ThaiTokenizer


# Thai example sentences taken from the PyThaiNLP Tutorial:
#   https://www.thainlp.org/pythainlp/tutorials/notebooks/pythainlp_get_started.html#Word
@pytest.mark.parametrize(
    "msg,n,setting",
    it.product(
        ["ก็จะรู้ความชั่วร้ายที่ทำไว้     และคงจะไม่ยอมให้ทำนาบนหลังคน "],
        [14],
        [True, False],
    ),
)
def test_thai_tokenizer_length(msg, n, setting):
    """We should add the correct number of tokens."""
    message = Message({TEXT: msg})
    tok = ThaiTokenizer(config={"case_sensitive": setting})
    tok.process([message])
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert len(tokens) == n
