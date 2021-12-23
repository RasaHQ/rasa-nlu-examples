import pytest
import pathlib

from rasa.shared.nlu.training_data.message import Message
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer

from .nlu_sparse import TfIdfFeaturizer


@pytest.fixture
def tfidf_featurizer(tmpdir):
    """Generate a tfidf vectorizer with a tmpdir as the model storage."""
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)
    return TfIdfFeaturizer(
        config=TfIdfFeaturizer.get_default_config(),
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )


tokeniser = WhitespaceTokenizer(
    {
        "only_alphanum": False,
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_",
    }
)


@pytest.mark.parametrize(
    "text, expected", [("hello", 1), ("hello world", 2), ("hello there world", 3)]
)
def test_sparse_feats_added(tfidf_featurizer, text, expected):
    """Checks if the sizes are appropriate."""
    # Create a message
    msg = Message({"text": text})

    # Process will process a list of Messages
    tokeniser.process([msg])
    tfidf_featurizer.train(TrainingData([msg]))
    tfidf_featurizer.process([msg])
    # Check that the message has been processed correctly
    seq_feats, sent_feats = msg.get_sparse_features("text")
    print(seq_feats.features.shape, sent_feats.features.shape)
    # We should have a feature per token
    assert seq_feats.features.shape[0] == expected
    # Sentence features should be have single row of data
    assert sent_feats.features.shape[0] == 1
