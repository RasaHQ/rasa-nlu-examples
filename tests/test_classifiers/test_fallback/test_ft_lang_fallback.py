import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa_nlu_examples.classifiers import FasttextLanguageFallbackClassifier
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.message import Message

config_man = dict(
    lang="en",
    vs=1000,
    dim=25,
    model_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.model",
    emb_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.d25.w2v.bin",
)
tokenizer = WhitespaceTokenizer()


@pytest.mark.parametrize(
    "txt,lang", [("ik spreek een taal", "nl"), ("je parle une langue", "fr")]
)
def test_detect_obvious_cases(txt, lang):
    config_dict = {
        "language": "en",
        "threshold": 0.7,
        "min_tokens": 2,
        "min_chars": 8,
        "intent_triggered": "non_english",
        "cache_dir": "tests/data/fasttext",
        "file": "lid.176.ftz",
    }
    ft_lang = FasttextLanguageFallbackClassifier(config_dict)
    message = Message({TEXT: txt})
    tokenizer.process(message=message)
    ft_lang.process(message=message)
    assert message.get(INTENT) == "non_english"


@pytest.mark.parametrize(
    "txt,lang", [("i am speaking english", "en"), ("this too should pass", "en")]
)
def test_no_change_on_english(txt, lang):
    config_dict = {
        "language": "en",
        "threshold": 0.3,
        "min_tokens": 2,
        "min_chars": 8,
        "intent_triggered": "non_english",
        "cache_dir": "tests/data/fasttext",
        "file": "lid.176.ftz",
    }
    ft_lang = FasttextLanguageFallbackClassifier(config_dict)
    message = Message({TEXT: txt, INTENT: "assigned_before"})
    tokenizer.process(message=message)
    ft_lang.process(message=message)
    assert message.get(INTENT) == "assigned_before"
