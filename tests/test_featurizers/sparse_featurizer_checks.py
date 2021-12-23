from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.training_data.message import Message
import itertools as it


def test_component_requires_tokenizer(tokenizer, featurizer, msg):
    """The component should have a tokenizer in its required components."""
    req_components = featurizer.__class__.required_components()
    predicate = [c == Tokenizer for c in req_components]
    assert any(predicate)


def test_component_no_features_on_no_tokens(tokenizer, featurizer, msg):
    """The component does not set any sparse features if there are no tokens."""
    message = Message({TEXT: msg})
    featurizer.process([message])
    seq_vecs, sen_vecs = message.get_sparse_features(TEXT, [])
    assert not seq_vecs
    assert not sen_vecs


def test_component_adds_features(tokenizer, featurizer, msg):
    """If there are no features we need to add them"""
    message = Message({TEXT: msg})
    tokenizer.process([message])
    tokens = message.get(TOKENS_NAMES[TEXT])

    featurizer.process([message])
    seq_vecs, sen_vecs = message.get_sparse_features(TEXT, [])
    assert seq_vecs.features.shape[0] == len(tokens)
    assert sen_vecs.features.shape[0] == 1


def test_component_does_not_remove_features(tokenizer, featurizer, msg):
    """If there are features we need to add not remove them"""
    message = Message({TEXT: msg})
    tokenizer.process([message])
    featurizer.process([message])
    seq_vecs1, sen_vecs1 = message.get_sparse_features(TEXT, [])

    featurizer.process([message])
    seq_vecs2, sen_vecs2 = message.get_sparse_features(TEXT, [])

    assert (seq_vecs1.features.shape[1] * 2) == seq_vecs2.features.shape[1]
    assert (sen_vecs1.features.shape[1] * 2) == sen_vecs2.features.shape[1]


sparse_feature_checks = (
    test_component_adds_features,
    test_component_does_not_remove_features,
    test_component_no_features_on_no_tokens,
    test_component_requires_tokenizer,
)


def sparse_standard_test_combinations(
    tokenizer, featurizer, messages=None, feature_checks=sparse_feature_checks
):
    if not messages:
        messages = [
            "hello",
            "hello there",
            "hello there again",
            "this is quite interesting",
            "dude",
            "foo",
            "bar",
            "buzz",
        ]
    return it.product([f for f in feature_checks], [tokenizer], [featurizer], messages)
