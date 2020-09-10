import pytest

from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT, TOKENS_NAMES, SPARSE_FEATURE_NAMES
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa_nlu_examples.tokenizers.stanzatokenizer import StanzaTokenizer


@pytest.mark.parametrize(
    "msg,n", zip(["hello", "hello there", "hello there vincent"], [1, 2, 3])
)
def test_stanza_correct_length(msg, n):
    """We should add the correct number of tokens."""
    message = Message(msg)
    tok = StanzaTokenizer(
        component_config={"lang": "en", "cache_dir": "tests/data/stanza"}
    )
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    # We also generate a __CLS__ token
    assert len(tokens) == n + 1


def test_stanza_lemma():
    """We need to attach correct lemmas"""
    message = Message("i am running and giving many greetings")
    tok = StanzaTokenizer(
        component_config={"lang": "en", "cache_dir": "tests/data/stanza"}
    )
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
    tok = StanzaTokenizer(
        component_config={"lang": "en", "cache_dir": "tests/data/stanza"}
    )
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


def fetch_sparse_features(txt, tokenizer, featurizer):
    message = Message("my advices include to give advice and giving many greetings")
    tokenizer.process(message)
    featurizer.train(TrainingData([message]))
    featurizer.process(message)
    return message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()


def test_component_changes_features_cvf():
    """If there are no features we need to add them"""
    tok_whitespace = WhitespaceTokenizer()
    tok_stanza = StanzaTokenizer(
        component_config={"lang": "en", "cache_dir": "tests/data/stanza"}
    )
    txt = "i am running and giving many greetings"
    feats_whitespace = fetch_sparse_features(
        txt, tok_whitespace, CountVectorsFeaturizer()
    )
    feats_stanza = fetch_sparse_features(txt, tok_stanza, CountVectorsFeaturizer())

    # Because the lemma is being used internally we expect less features
    assert feats_stanza.shape[1] < feats_whitespace.shape[1]


def test_component_changes_features_lex():
    """If there are no features we need to add them"""
    tok_whitespace = WhitespaceTokenizer()
    tok_stanza = StanzaTokenizer(
        component_config={"lang": "en", "cache_dir": "tests/data/stanza"}
    )
    txt = "i am running and giving many greetings"
    feats = {
        "features": [
            ["low", "title", "upper", "pos"],
            ["BOS", "EOS", "low", "upper", "title", "digit", "pos"],
            ["low", "title", "upper", "pos"],
        ]
    }
    feats_whitespace = fetch_sparse_features(
        txt, tok_whitespace, LexicalSyntacticFeaturizer(feats)
    )
    feats_stanza = fetch_sparse_features(
        txt, tok_stanza, LexicalSyntacticFeaturizer(feats)
    )

    # Because the part of speech tags are now used we'd expect stanza to have more features here.
    assert feats_stanza.shape[1] > feats_whitespace.shape[1]
