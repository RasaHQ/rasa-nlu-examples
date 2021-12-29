import pytest
from rasa_nlu_examples.tokenizers import BlankSpacyTokenizer

from rasa.shared.nlu.constants import TEXT
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.training_data.message import Message


examples = [
    {
        "text": "i am running and giving many greetings",
        "lang": "en",
        "result": ["i", "am", "running", "and", "giving", "many", "greetings"],
    },
    {
        "text": "कृतिदेव से यूनिकोड फॉन्ट कन्वर्शन",
        "lang": "hi",
        "result": ["कृतिदेव", "से", "यूनिकोड", "फॉन्ट", "कन्वर्शन"],
    },
]


@pytest.mark.parametrize("example", examples)
def test_base_examples(example):
    message = Message({TEXT: example["text"]})
    tok = BlankSpacyTokenizer(config={"lang": example["lang"]})
    tok.process([message])
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert [t.text for t in tokens] == example["result"]
