import pytest

from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES, TOKENS_NAMES
from rasa.nlu.training_data import Message
import itertools as it


def test_component_raises_error_no_tokens(tokenizer, featurizer, msg):
    """The component needs to throw an error if there are no tokens."""
    # we expect a tokenizer to be a required component
    req_components = featurizer.__class__.required_components()
    predicate = [c == Tokenizer for c in req_components]
    assert any(predicate)

    # we expect an error to occur here
    message = Message(msg)
    with pytest.raises(KeyError):
        featurizer.process(message)


def test_component_adds_features(tokenizer, featurizer, msg):
    """If there are no features we need to add them"""
    message = Message(msg)
    tokenizer.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])

    featurizer.process(message)
    vectors = message.get(DENSE_FEATURE_NAMES[TEXT])
    assert vectors.shape[0] == len(tokens)


def test_component_does_not_remove_features(tokenizer, featurizer, msg):
    """If there are features we need to add not remove them"""
    message = Message(msg)
    tokenizer.process(message)
    featurizer.process(message)
    first_vectors = message.get(DENSE_FEATURE_NAMES[TEXT])

    featurizer.process(message)
    second_vectors = message.get(DENSE_FEATURE_NAMES[TEXT])

    assert (first_vectors.shape[1] * 2) == second_vectors.shape[1]


dense_feature_checks = (
    test_component_adds_features,
    test_component_does_not_remove_features,
    test_component_raises_error_no_tokens,
)


def dense_standard_test_combinations(
    tokenizer, featurizer, messages=None, feature_checks=dense_feature_checks
):
    if not messages:
        messages = [
            "",
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
