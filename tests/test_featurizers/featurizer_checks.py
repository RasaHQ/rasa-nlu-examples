import numpy as np
import pytest

from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.training_data import TrainingData
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES, TOKENS_NAMES, RESPONSE, INTENT
from rasa.nlu.training_data import Message
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


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
    tokens = tokenizer.tokenize(message, attribute=TEXT)
    tokens = Tokenizer.add_cls_token(tokens, attribute=TEXT)
    message.set(TOKENS_NAMES[TEXT], tokens)

    featurizer.process(message)
    vectors = message.get(DENSE_FEATURE_NAMES[TEXT])
    print(vectors.shape)
    assert vectors.shape[0] == len(tokens)


def test_component_does_not_remove_features(tokenizer, featurizer, msg):
    """If there are no features we need to add them"""
    message = Message(msg)
    tokens = tokenizer.tokenize(message, attribute=TEXT)
    tokens = Tokenizer.add_cls_token(tokens, attribute=TEXT)
    message.set(TOKENS_NAMES[TEXT], tokens)
    featurizer.process(message)
    first_vectors = message.get(DENSE_FEATURE_NAMES[TEXT])

    featurizer.process(message)
    second_vectors = message.get(DENSE_FEATURE_NAMES[TEXT])

    assert (first_vectors.shape[1] * 2) == second_vectors.shape[1]


dense_feature_checks = (
    test_component_adds_features,
    test_component_does_not_remove_features,
    test_component_raises_error_no_tokens
)