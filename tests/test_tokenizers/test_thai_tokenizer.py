import pytest

import numpy as np

from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa_nlu_examples.tokenizers import ThaiTokenizer


# Thai example sentences taken from the PyThaiNLP Tutorial:
#   https://www.thainlp.org/pythainlp/tutorials/notebooks/pythainlp_get_started.html#Word
@pytest.mark.parametrize(
    "msg,n",
    zip(["ก็จะรู้ความชั่วร้ายที่ทำไว้     และคงจะไม่ยอมให้ทำนาบนหลังคน "], [14]),
)
def test_thai_tokenizer_length(msg, n):
    """We should add the correct number of tokens."""
    message = Message({TEXT: msg})
    tok = ThaiTokenizer()
    tok.process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])
    assert len(tokens) == n


def fetch_sparse_features(txt, tokenizer, featurizer):

    message = Message({TEXT: txt})
    tokenizer.process(message)
    featurizer.train(TrainingData([message]))
    featurizer.process(message)

    seq_vecs, sen_vecs = message.get_sparse_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    return seq_vecs.toarray()


def test_component_changes_features_cvf():
    """If there are no features we need to add them"""
    tokenizer = ThaiTokenizer()

    txt = "ก็จะรู้ความชั่วร้ายที่ทำไว้     และคงจะไม่ยอมให้ทำนาบนหลังคน "
    feats = fetch_sparse_features(
        txt=txt, tokenizer=tokenizer, featurizer=CountVectorsFeaturizer()
    )

    assert feats.shape[1] > 0 and isinstance(feats, np.ndarray)
