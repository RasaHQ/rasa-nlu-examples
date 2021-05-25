import pathlib

import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from .dense_featurizer_checks import dense_standard_test_combinations
from rasa_nlu_examples.featurizers.dense.fasttext_featurizer import FastTextFeaturizer

test_folder = pathlib.Path(__file__).parent.parent.absolute()
cache_dir = str(test_folder / "data" / "fasttext")
file_name = "custom_fasttext_model.bin"

config = {"cache_dir": cache_dir, "file": file_name}

tokenizer = WhitespaceTokenizer()
featurizer = FastTextFeaturizer(component_config=config)


@pytest.mark.fasttext
def test_model_loaded():
    assert featurizer


@pytest.mark.fasttext
@pytest.mark.parametrize(
    "test_fn,tok,feat,msg",
    dense_standard_test_combinations(tokenizer=tokenizer, featurizer=featurizer),
)
def test_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)


@pytest.mark.fasttext
def test_raise_cachedir_not_given_error():
    with pytest.raises(ValueError):
        FastTextFeaturizer(component_config={"file": "foobar.kv"})


@pytest.mark.fasttext
def test_raise_file_not_given_error():
    with pytest.raises(ValueError):
        FastTextFeaturizer(component_config={"cache_dir": "some/path"})


@pytest.mark.fasttext
def test_raise_cachedir_error():
    bad_folder = str(test_folder / "foobar")
    with pytest.raises(FileNotFoundError):
        FastTextFeaturizer(
            component_config={"cache_dir": bad_folder, "file": file_name}
        )


@pytest.mark.fasttext
def test_raise_file_error():
    with pytest.raises(FileNotFoundError):
        FastTextFeaturizer(
            component_config={"cache_dir": test_folder, "file": "dinosaur.bin"}
        )
