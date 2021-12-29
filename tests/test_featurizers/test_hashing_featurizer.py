from typing import Callable, List, Text

import pytest
import scipy.sparse
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.message import Message
from rasa_nlu_examples.featurizers.sparse.hashing_featurizer import HashingFeaturizer
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext

from .sparse_featurizer_checks import sparse_standard_test_combinations

node_storage = LocalModelStorage("tmp/storage")
node_resource = Resource("tokenizer")
context = ExecutionContext(node_storage, node_resource)
config = dict(n_features=1024, norm=None, alternate_sign=False)


@pytest.mark.parametrize(
    "test_fn,tokenizer,featurizer,messages",
    sparse_standard_test_combinations(
        tokenizer=WhitespaceTokenizer(config=WhitespaceTokenizer.get_default_config()),
        featurizer=HashingFeaturizer(config=config, name=context.node_name),
    ),
)
def test_auto_featurizer_checks(
    test_fn: Callable,
    tokenizer: Tokenizer,
    featurizer: Featurizer,
    messages: List[Text],
):
    test_fn(tokenizer, featurizer, messages)


@pytest.fixture
def whitespace_tokenizer() -> WhitespaceTokenizer:
    return WhitespaceTokenizer(config=WhitespaceTokenizer.get_default_config())


@pytest.fixture
def hashing_featurizer() -> HashingFeaturizer:
    return HashingFeaturizer(config=config, name=context.node_name)


def test_features_are_sparse(
    whitespace_tokenizer: WhitespaceTokenizer,
    hashing_featurizer: HashingFeaturizer,
):
    message = Message.build("am I talking to a bot")

    whitespace_tokenizer.process([message])
    hashing_featurizer.process([message])

    for feature in message.features:
        assert scipy.sparse.issparse(feature.features)


def test_feature_shapes(
    whitespace_tokenizer: WhitespaceTokenizer,
    hashing_featurizer: HashingFeaturizer,
):
    text = "am I talking to a bot"
    message = Message.build(text)

    whitespace_tokenizer.process([message])
    hashing_featurizer.process([message])

    feat_tok, feat_sent = message.get_sparse_features("text")
    assert feat_tok.features.shape == (
        len(text.split()),
        config["n_features"],
    )
    assert feat_sent.features.shape == (1, config["n_features"])
    assert feat_tok.features.sum() == feat_sent.features.sum()


def test_feature_overlap(
    whitespace_tokenizer: WhitespaceTokenizer,
    hashing_featurizer: HashingFeaturizer,
):
    message_1 = Message.build("am I talking to a bot")
    whitespace_tokenizer.process([message_1])
    hashing_featurizer.process([message_1])

    message_2 = Message.build("sharing a subset of words")
    whitespace_tokenizer.process([message_2])
    hashing_featurizer.process([message_2])

    message_3 = Message.build("completely different message")
    whitespace_tokenizer.process([message_3])
    hashing_featurizer.process([message_3])

    _, feat_sent_1 = message_1.get_sparse_features("text")
    _, feat_sent_2 = message_2.get_sparse_features("text")
    _, feat_sent_3 = message_3.get_sparse_features("text")

    # the first ans second message have overlapping vocabulary, so some column
    # index should contain non-zero values in both
    assert feat_sent_1.features.dot(feat_sent_2.features.T) > 0
    assert feat_sent_1.features.dot(feat_sent_3.features.T) == 0


def test_char_wb_analyzer(
    whitespace_tokenizer: WhitespaceTokenizer,
):
    config = dict(n_features=1024, norm=None, alternate_sign=False)
    config["analyzer"] = "char_wb"  # extract n-grams within word boundaries
    hashing_featurizer = HashingFeaturizer(config=config, name=context.node_name)

    text = "am I talking to a bot"
    message = Message.build(text)

    whitespace_tokenizer.process([message])
    hashing_featurizer.process([message])

    feat_tok, feat_sent = message.get_sparse_features("text")

    # Each token (as split by WhitespaceTokenizer) is mapped to a vector, so the number
    # of rows is still the same as with the `word` analyzer. However, the vectors are
    # built by mapping each character n-gram to a column index, so each row will contain one
    # entry per character n-gram. Here we use unigrams only, since `ngram_range=(1, 1)`.
    assert feat_tok.features.shape == (
        len(text.split()),
        config["n_features"],
    )
    for row_idx, token in enumerate(text.split()):
        assert (
            feat_tok.features.tocsr()[row_idx].sum() == len(token) + 2
        )  # words are padded with spaces on each side

    assert feat_sent.features.shape == (1, config["n_features"])
    assert feat_tok.features.sum() == feat_sent.features.sum()
