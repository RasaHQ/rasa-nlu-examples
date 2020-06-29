import itertools as it
import pathlib

import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from rasa_nlu_examples.featurizers.dense import BytePairFeaturizer
from .featurizer_checks import dense_feature_checks

test_folder = pathlib.Path(__file__).parent.parent.absolute()
cache_dir = str(test_folder / "data")
file_name = "custom_fasttext_model.bin"

config = dict(lang="en", vs=1000, dim=25, vs_fallback=True)
tokenizer = WhitespaceTokenizer()
featurizer = BytePairFeaturizer(component_config=config)

combinations = it.product(
    [f for f in dense_feature_checks],
    [tokenizer],
    [featurizer],
    ["", "hello", "hello there", "hello there again", "this is quite interesting", "dude", "foo", "bar", "buzz"]
)


def test_model_loaded():
    assert featurizer


@pytest.mark.parametrize("test_fn,tok,feat,msg", combinations)
def test_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)
