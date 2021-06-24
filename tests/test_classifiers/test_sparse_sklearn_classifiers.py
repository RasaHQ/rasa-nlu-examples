import copy
from typing import Text, Type, Optional

import pytest
import scipy.sparse
import numpy as np

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    INTENT_NAME_KEY,
    FEATURE_TYPE_SENTENCE,
    PREDICTED_CONFIDENCE_KEY,
)

from rasa_nlu_examples.classifiers.sparse_naive_bayes_intent_classifier import (
    SparseNaiveBayesIntentClassifier,
)
from rasa_nlu_examples.classifiers.sparse_logistic_regression_intent_classifier import (
    SparseLogisticRegressionIntentClassifier,
)
from rasa_nlu_examples.classifiers.sparse_sklearn_intent_classifier import (
    SparseSklearnIntentClassifier,
)

SPARSE_SKLEARN_INTENT_CLASSIFIERS = [
    SparseNaiveBayesIntentClassifier,
    SparseLogisticRegressionIntentClassifier,
]


def generate_dummy_training_data(
    num_messages: int = 10,
    with_dense_features: bool = False,
    seed: Optional[int] = None,
) -> TrainingData:
    """creates some minimal dummy TrainingData that contains intent training
    examples with random features, which are sparse by default.

    :param with_dense_features: if set to True, some dense features will be added
    :param num_messages: number of training_examples to be created
    :param seed: seed for the rng used to create random features
    """
    rng = np.random.default_rng(seed=seed)

    def random_feature(sparse: bool = True):
        arr = rng.integers(3, size=(10))
        return Features(
            features=(scipy.sparse.coo_matrix(arr) if sparse else arr),
            feature_type=FEATURE_TYPE_SENTENCE,
            origin="",
            attribute=TEXT,
        )

    training_examples = [
        Message(
            data={
                TEXT: "text",  # needed, otherwise not recognised as NLU example
                INTENT: intent,
            },
            features=[random_feature(sparse=True)]
            + ([random_feature(sparse=False)] if with_dense_features else []),
        )
        for intent in ["intent1", "intent2"]
        for _ in range(num_messages)
    ]
    return TrainingData(training_examples=training_examples)


@pytest.fixture(scope="module")
def dummy_data():
    return generate_dummy_training_data(seed=2345)


@pytest.fixture(scope="module")
def dummy_data_with_dense_features():
    return generate_dummy_training_data(seed=2345, with_dense_features=True)


@pytest.mark.parametrize("cls,", SPARSE_SKLEARN_INTENT_CLASSIFIERS)
def test_creation_of_sklearn_classifier_with_defaults(
    cls: Type[SparseSklearnIntentClassifier],
):
    intent_classifier = cls()
    sklearn_clf = intent_classifier.create_sklearn_classifier(**cls.defaults)
    assert sklearn_clf


@pytest.mark.parametrize("cls,", SPARSE_SKLEARN_INTENT_CLASSIFIERS)
def test_training_and_inference_on_dummy_data(
    dummy_data: TrainingData,
    cls: Type[SparseSklearnIntentClassifier],
):
    # train
    model = cls()
    training_data = dummy_data
    model.train(training_data=training_data)
    assert model.clf is not None
    assert model.le is not None
    # predict (needs train)
    dummy_message = training_data.intent_examples[0]
    _, sparse_sentence_features = dummy_message.get_sparse_features(TEXT)
    sparse_matrix = sparse_sentence_features.features
    indices, probabilities = model.predict(sparse_matrix)
    assert len(set(indices[0])) == len(model.clf.classes_)
    assert np.isclose(np.sum(probabilities[0]), 1.0)
    # process (uses train)
    dummy_message = copy.deepcopy(
        training_data.intent_examples[0]
    )  # deepcopy to avoid side-effects
    model.process(dummy_message)
    intent = dummy_message.get(INTENT)
    assert intent is not None
    assert isinstance(intent.get(INTENT_NAME_KEY), Text)
    assert isinstance(intent.get(PREDICTED_CONFIDENCE_KEY), float)
    confidence = intent.get(PREDICTED_CONFIDENCE_KEY)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    # process (missing features / empty message)
    dummy_message = Message()
    with pytest.raises(ValueError, match="No sparse sentence features present"):
        model.process(dummy_message)
    assert dummy_message.get(INTENT) is None


@pytest.mark.parametrize("cls,", SPARSE_SKLEARN_INTENT_CLASSIFIERS)
def test_training_and_inference_on_empty_data(
    cls: Type[SparseSklearnIntentClassifier],
):
    # train on empty data
    model = cls()
    training_data = TrainingData()
    with pytest.warns(UserWarning, match="there are not enough intents"):
        model.train(training_data=training_data)
    assert model.clf is None
    assert model.le is None
    # process  (missing features / empty message)
    dummy_message = Message()
    model.process(dummy_message)
    intent = dummy_message.get(INTENT)
    assert intent is None


@pytest.mark.parametrize("cls", SPARSE_SKLEARN_INTENT_CLASSIFIERS)
def test_warn_on_dense_features(
    dummy_data_with_dense_features: TrainingData,
    cls: Type[SparseSklearnIntentClassifier],
):
    model = cls()
    with pytest.warns(
        UserWarning, match="Dense features are being computed but not used"
    ):
        model.train(training_data=dummy_data_with_dense_features)
