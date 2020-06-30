import pathlib

import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from .featurizer_checks import dense_standard_test_combinations
from rasa_nlu_examples.featurizers.dense.fasttext import FastTextFeaturizer

test_folder = pathlib.Path(__file__).parent.parent.absolute()
cache_dir = str(test_folder / "data")
file_name = "custom_fasttext_model.bin"

config = {"cache_dir": cache_dir, "file": file_name}

tokenizer = WhitespaceTokenizer()
featurizer = FastTextFeaturizer(component_config=config)


def test_model_loaded():
    assert featurizer


@pytest.mark.parametrize(
    "test_fn,tok,feat,msg",
    dense_standard_test_combinations(tokenizer=tokenizer, featurizer=featurizer),
)
def test_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)
