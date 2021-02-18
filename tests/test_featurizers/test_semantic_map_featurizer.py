import pathlib

import pytest
import scipy.sparse
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE, TEXT
from rasa.shared.nlu.training_data.message import Message
import rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer
from rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer import (
    SemanticMapFeaturizer,
    SemanticFingerprint,
)

test_directory = pathlib.Path(__file__).parent.parent.absolute()
example_semantic_map = (
    test_directory / "data" / "semantic_map" / "dummy_semantic_map.json"
)


def test_features_are_sparse():
    tokenizer = WhitespaceTokenizer()
    featurizer = SemanticMapFeaturizer(
        {"pretrained_semantic_map": str(example_semantic_map), "pooling": "merge"}
    )
    message = Message.build("word1 word3")

    tokenizer.process(message)
    featurizer.process(message)

    for feature in message.features:
        assert scipy.sparse.issparse(feature.features)


def test_feature_shapes():
    tokenizer = WhitespaceTokenizer()
    featurizer = SemanticMapFeaturizer(
        {"pretrained_semantic_map": str(example_semantic_map), "pooling": "merge"}
    )
    message = Message.build("word1 word3")

    tokenizer.process(message)
    featurizer.process(message)

    for feature in message.features:
        assert (
            feature.type == FEATURE_TYPE_SEQUENCE and feature.features.shape == (2, 37)
        ) or (
            feature.type == FEATURE_TYPE_SENTENCE and feature.features.shape == (1, 37)
        )


def test_no_features_on_no_tokens():
    """The component does not set any dense features if there are no tokens."""
    tokenizer = WhitespaceTokenizer()
    featurizer = SemanticMapFeaturizer(
        {"pretrained_semantic_map": str(example_semantic_map), "pooling": "merge"}
    )
    message = Message.build("word1 word3")

    # We skip: tokenizer.process(message)
    featurizer.process(message)

    seq_vecs, sen_vecs = message.get_sparse_features(TEXT, [])
    assert not seq_vecs
    assert not sen_vecs


def test_semantic_overlap():
    fp1 = SemanticFingerprint(1, 8, {1, 2, 3})
    fp2 = SemanticFingerprint(1, 8, {3, 5})
    assert (
        rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer.semantic_overlap(
            fp1, fp2, method="Jaccard"
        )
        == 1 / 4
    )
    assert (
        rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer.semantic_overlap(
            fp1, fp2, method="SzymkiewiczSimpson"
        )
        == 1 / 2
    )
    assert (
        rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer.semantic_overlap(
            fp1, fp2, method="Rand"
        )
        == 5 / 8
    )
