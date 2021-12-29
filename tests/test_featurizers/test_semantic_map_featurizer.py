"""
These tests have been commented away. It's a bit unclear to what extend we want
to continue supporting this feature, hence it was ignored during the 2.x -> 3.x port.
"""
# from pathlib import Path

# import pytest
# import rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer
# import scipy.sparse
# from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
# from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE, TEXT
# from rasa.shared.nlu.training_data.message import Message
# from rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer import (
#     SemanticFingerprint,
#     SemanticMap,
#     SemanticMapFeaturizer,
# )

# test_directory = Path(__file__).parent.parent.absolute()


# @pytest.fixture
# def semantic_map_file() -> Path:
#     return test_directory / "data" / "semantic_map" / "dummy_semantic_map.json"


# @pytest.fixture
# def whitespace_tokenizer() -> WhitespaceTokenizer:
#     return WhitespaceTokenizer()


# @pytest.fixture
# def semantic_map_featurizer(semantic_map_file) -> SemanticMapFeaturizer:
#     return SemanticMapFeaturizer(
#         {"pretrained_semantic_map": str(semantic_map_file), "pooling": "merge"}
#     )


# def test_features_are_sparse(
#     whitespace_tokenizer: WhitespaceTokenizer,
#     semantic_map_featurizer: SemanticMapFeaturizer,
# ):
#     message = Message.build("word1 word3")

#     whitespace_tokenizer.process(message)
#     semantic_map_featurizer.process(message)

#     for feature in message.features:
#         assert scipy.sparse.issparse(feature.features)


# def test_feature_shapes(
#     whitespace_tokenizer: WhitespaceTokenizer,
#     semantic_map_featurizer: SemanticMapFeaturizer,
# ):
#     message = Message.build("word1 word3")

#     whitespace_tokenizer.process(message)
#     semantic_map_featurizer.process(message)

#     for feature in message.features:
#         assert (
#             feature.type == FEATURE_TYPE_SEQUENCE and feature.features.shape == (2, 37)
#         ) or (
#             feature.type == FEATURE_TYPE_SENTENCE and feature.features.shape == (1, 37)
#         )


# def test_no_features_on_no_tokens(semantic_map_featurizer: SemanticMapFeaturizer):
#     """The component does not set any sparse features if tokens are not available."""
#     message = Message.build("word1 word3")

#     # We skip: whitespace_tokenizer.process(message)
#     semantic_map_featurizer.process(message)

#     seq_vecs, sen_vecs = message.get_sparse_features(TEXT, [])
#     assert not seq_vecs
#     assert not sen_vecs


# def test_semantic_overlap():
#     fp1 = SemanticFingerprint(1, 8, {1, 2, 3})
#     fp2 = SemanticFingerprint(1, 8, {3, 5})
#     overlap = (
#         rasa_nlu_examples.featurizers.sparse.semantic_map_featurizer.semantic_overlap
#     )
#     assert overlap(fp1, fp2, method="Jaccard") == 1 / 4
#     assert overlap(fp1, fp2, method="SzymkiewiczSimpson") == 1 / 2
#     assert overlap(fp1, fp2, method="Rand") == 5 / 8


# def test_semantic_merge_does_not_activate_inactive_cells(semantic_map_file: Path):
#     smap = SemanticMap(semantic_map_file)

#     fp1 = smap.get_term_fingerprint("word1")
#     fp2 = smap.get_term_fingerprint("word2")
#     fp3 = smap.get_term_fingerprint("word3")
#     all_activations = set.union(
#         fp1.as_activations(), fp2.as_activations(), fp3.as_activations()
#     )

#     merged_fp = smap.get_fingerprint("word1 word2 word3")
#     merged_activations = merged_fp.as_activations()

#     assert merged_activations.issubset(all_activations)


# def test_error_when_file_missing():
#     with pytest.raises(FileNotFoundError):
#         SemanticMapFeaturizer({"pretrained_semantic_map": ":{*(^$%YBHJKI&T^"})
#     with pytest.raises(FileNotFoundError):
#         SemanticMapFeaturizer({"pretrained_semantic_map": "./nonexistent.json"})
