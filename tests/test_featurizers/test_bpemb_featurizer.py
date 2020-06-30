import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from rasa_nlu_examples.featurizers.dense import BytePairFeaturizer
from .featurizer_checks import dense_standard_test_combinations


config = dict(lang="en", vs=1000, dim=25, vs_fallback=True)
tokenizer = WhitespaceTokenizer()
featurizer = BytePairFeaturizer(component_config=config)


def test_model_loaded():
    assert featurizer


@pytest.mark.parametrize(
    "test_fn,tok,feat,msg",
    dense_standard_test_combinations(tokenizer=tokenizer, featurizer=featurizer),
)
def test_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)
