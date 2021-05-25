import pytest

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa_nlu_examples.tokenizers.stanzatokenizer import StanzaTokenizer


@pytest.mark.skip(reason="The Stanza project seems deprecated.")
@pytest.mark.parametrize(
    "msg,n", zip(["hello", "hello there", "hello there vincent"], [1, 2, 3])
)
def test_stanza_correct_length(msg, n):
    """We should add the correct number of tokens."""
    message = Message({TEXT: msg})
    tok = StanzaTokenizer(
        component_config={"lang": "en", "cache_dir": "tests/data/stanza"}
    )
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert len(tokens) == n


@pytest.mark.skip(reason="The Stanza project seems deprecated.")
def test_stanza_lemma():
    """We need to attach correct lemmas"""
    message = Message({TEXT: "i am running and giving many greetings"})
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
    ]


@pytest.mark.skip(reason="The Stanza project seems deprecated.")
def test_stanza_pos():
    """We need to attach correct POS"""
    message = Message({TEXT: "i am running and giving many greetings"})
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
    ]


@pytest.mark.skip(reason="The Stanza project seems deprecated.")
def fetch_sparse_features(txt, tokenizer, featurizer):
    message = Message(
        {TEXT: "my advices include to give advice and giving many greetings"}
    )
    tokenizer.process(message)
    featurizer.train(TrainingData([message]))
    featurizer.process(message)
    seq_vecs, sen_vecs = message.get_sparse_features(TEXT, [])
    return seq_vecs.features.toarray()


@pytest.mark.skip(reason="The Stanza project seems deprecated.")
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


@pytest.mark.skip(reason="The Stanza project seems deprecated.")
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
