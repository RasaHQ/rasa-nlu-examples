import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from rasa_nlu_examples.featurizers.dense import BytePairFeaturizer
from .featurizer_checks import dense_standard_test_combinations

config_man = dict(
    lang="en",
    vs=1000,
    dim=25,
    model_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.model",
    emb_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.d25.w2v.bin",
)
config_auto = dict(lang="en", vs=1000, dim=25, vs_fallback=True)
tokenizer = WhitespaceTokenizer()


@pytest.mark.parametrize(
    "test_fn,tok,feat,msg",
    dense_standard_test_combinations(
        tokenizer=tokenizer, featurizer=BytePairFeaturizer(component_config=config_auto)
    ),
)
def test_auto_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)


@pytest.mark.parametrize(
    "test_fn,tok,feat,msg",
    dense_standard_test_combinations(
        tokenizer=tokenizer, featurizer=BytePairFeaturizer(component_config=config_man)
    ),
)
def test_man_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)


def test_raise_error_missing_model_file():
    config_bad = dict(
        lang="en",
        vs=1000,
        dim=25,
        model_file="tests/data/bytepair/en/en.dinosaur.bpe.vs1000.model",
        emb_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.d25.w2v.bin",
    )
    with pytest.raises(FileNotFoundError):
        BytePairFeaturizer(component_config=config_bad)


def test_raise_error_missing_emb_file():
    config_bad = dict(
        lang="en",
        vs=1000,
        dim=25,
        model_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.model",
        emb_file="tests/data/bytepair/en/en.wiki.dinosaur.vs1000.d25.w2v.bin",
    )
    with pytest.raises(FileNotFoundError):
        BytePairFeaturizer(component_config=config_bad)


def test_config_missing():
    config_bad = dict(
        lang="en",
        vs=1000,
        dim=25,
        model_file="tests/data/bytepair/en/en.wiki.bpe.vs1000.model",
        emb_file="tests/data/bytepair/en/en.wiki.dinosaur.vs1000.d25.w2v.bin",
    )
    with pytest.raises(FileNotFoundError):
        BytePairFeaturizer(component_config=config_bad)


@pytest.mark.parametrize(
    "conf", [dict(lang="en", vs=1000), dict(lang="en", dim=25), dict(dim=25, vs=1000)]
)
def test_raise_missing_error(conf):
    with pytest.raises(ValueError):
        BytePairFeaturizer(component_config=conf)
