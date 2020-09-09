import pytest

import numpy as np

from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT, TOKENS_NAMES, SPARSE_FEATURE_NAMES
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa_nlu_examples.tokenizers import ThaiTokenizer

# Thai example sentences taken from the PyThaiNLP Tutorial:
#   https://www.thainlp.org/pythainlp/tutorials/notebooks/pythainlp_get_started.html#Word
@pytest.mark.parametrize(
    "msg,n", zip(["ก็จะรู้ความชั่วร้ายที่ทำไว้     และคงจะไม่ยอมให้ทำนาบนหลังคน "], [14])
)
def test_thai_tokenizer_length(msg, n):
    """We should add the correct number of tokens."""
    message = Message(msg)
    tok = ThaiTokenizer()
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    # We also generate a __CLS__ token
    assert len(tokens) == n + 1


def fetch_sparse_features(txt, tokenizer, featurizer):

    message = Message(txt)
    tokenizer.process(message)
    featurizer.train(TrainingData([message]))
    featurizer.process(message)

    return message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()


def test_component_changes_features_cvf():
    """If there are no features we need to add them"""
    tokenizer = ThaiTokenizer()

    txt = "ก็จะรู้ความชั่วร้ายที่ทำไว้     และคงจะไม่ยอมให้ทำนาบนหลังคน "
    feats = fetch_sparse_features(txt=txt, tokenizer=tokenizer, featurizer=CountVectorsFeaturizer())

    assert feats.shape[1] > 0 and isinstance(feats, np.ndarray)