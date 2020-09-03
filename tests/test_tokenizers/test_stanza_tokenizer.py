import pytest
from rasa.nlu.training_data import Message
from rasa.nlu.constants import TEXT, TOKENS_NAMES
from rasa_nlu_examples.tokenizers.stanzatokenizer import StanzaTokenizer


@pytest.mark.parametrize(
    "msg,n", zip(["hello", "hello there", "hello there vincent"], [1, 2, 3])
)
def test_stanza_correct_length(msg, n):
    """We should add the correct number of tokens."""
    message = Message(msg)
    tok = StanzaTokenizer(component_config={"lang": "en"})
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    # We also generate a __CLS__ token
    assert len(tokens) == n + 1


def test_stanza_lemma():
    """We need to attach correct lemmas"""
    message = Message("i am running and giving many greetings")
    tok = StanzaTokenizer(component_config={"lang": "en"})
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert [t.lemma for t in tokens] == [
        "i",
        "be",
        "run",
        "and",
        "give",
        "many",
        "greeting",
        "__CLS__",
    ]


def test_stanza_pos():
    """We need to attach correct POS"""
    message = Message("i am running and giving many greetings")
    tok = StanzaTokenizer(component_config={"lang": "en"})
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert [t.data.get("pos", "") for t in tokens] == [
        "PRON",
        "AUX",
        "VERB",
        "CCONJ",
        "VERB",
        "ADJ",
        "NOUN",
        "",
    ]
