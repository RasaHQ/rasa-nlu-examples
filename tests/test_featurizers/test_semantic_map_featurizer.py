import pathlib

import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.message import Message
from rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer import (
    SemanticMap,
    SemanticMapFeaturizer,
)

test_directory = pathlib.Path(__file__).parent.parent.absolute()
example_semantic_map = (
    test_directory / "data" / "semantic_map" / "dummy_semantic_map.json"
)


def test_features_are_sparse():
    tokenizer = WhitespaceTokenizer()
    smap_featurizer = SemanticMapFeaturizer(
        {"pretrained_semantic_map": str(example_semantic_map), "pooling": "merge"}
    )
    message = Message.build("word1 word3")

    tokenizer.process(message)
    smap_featurizer.process(message)

    for feature in message.features:
        assert isinstance(feature.features, scipy.sparse.coo_matrix)
