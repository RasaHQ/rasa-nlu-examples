import pathlib

import pytest
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from .dense_featurizer_checks import dense_standard_test_combinations
from rasa_nlu_examples.featurizers.dense.fasttext_featurizer import FastTextFeaturizer
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext


test_folder = pathlib.Path(__file__).parent.parent.absolute()
cache_path = str(test_folder / "data" / "fasttext" / "custom_fasttext_model.bin")
node_storage = LocalModelStorage("tmp/storage")
node_resource = Resource("tokenizer")
context = ExecutionContext(node_storage, node_resource)

config = {"cache_path": cache_path}

tokenizer = WhitespaceTokenizer(config=WhitespaceTokenizer.get_default_config())
featurizer = FastTextFeaturizer(config=config, name=context.node_name)


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
def test_raise_cachedir_not_exists():
    with pytest.raises(FileNotFoundError):
        FastTextFeaturizer(config={"cache_path": "foobar.kv"}, name=context.node_name)


@pytest.mark.fasttext
def test_raise_cachedir_not_given():
    with pytest.raises(ValueError):
        FastTextFeaturizer(config={}, name=context.node_name)
