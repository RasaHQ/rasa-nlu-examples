import pytest

from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext

from rasa_nlu_examples.featurizers.dense import BytePairFeaturizer
from .dense_featurizer_checks import dense_standard_test_combinations


node_storage = LocalModelStorage("tmp/storage")
node_resource = Resource("tokenizer")
context = ExecutionContext(node_storage, node_resource)

config_auto = dict(lang="en", vs=1000, dim=25, vs_fallback=True)
tokenizer = WhitespaceTokenizer(WhitespaceTokenizer.get_default_config())


@pytest.mark.parametrize(
    "test_fn,tok,feat,msg",
    dense_standard_test_combinations(
        tokenizer=tokenizer,
        featurizer=BytePairFeaturizer(config=config_auto, name=context.node_name),
    ),
)
def test_auto_featurizer_checks(test_fn, tok, feat, msg):
    test_fn(tok, feat, msg)


@pytest.mark.parametrize(
    "conf", [dict(lang="en", vs=1000), dict(lang="en", dim=25), dict(dim=25, vs=1000)]
)
def test_raise_missing_error(conf):
    with pytest.raises(ValueError):
        BytePairFeaturizer(config=conf, name=context.node_name)


@pytest.mark.parametrize(
    "text, expected", [("hello", 1), ("hello world", 2), ("hello there world", 3)]
)
def test_vs_size_manually(text, expected):
    """Checks if the sizes are appropriate."""
    config_man = dict(
        lang="en",
        vs=1000,
        dim=25,
    )
    bpemb_feat = BytePairFeaturizer(config=config_man, name=context.node_name)
    msg = Message({"text": text})

    # Process will process a list of Messages
    tokenizer.process([msg])
    bpemb_feat.process([msg])

    # Check that the message has been processed correctly
    seq_feats, sent_feats = msg.get_dense_features("text")
    assert seq_feats.features.shape == (expected, 25)
    assert sent_feats.features.shape == (1, 25)
