from abc import abstractmethod
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import numpy as np
import scipy.sparse
import sklearn

import rasa.shared.utils.io
import rasa.utils.io as io_utils
from rasa.shared.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_RANKING_KEY,
)

logger = logging.getLogger(__name__)


class SparseSklearnIntentClassifier(IntentClassifier):
    r"""Base class for intent classifiers using the sklearn framework with sparse features.
    Note that all sparse features will be used, i.e. there is no filtering for specific
    featurizers."""

    defaults = {}

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [SparseFeaturizer]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        clf: Optional[sklearn.base.ClassifierMixin] = None,
        le: Optional[sklearn.preprocessing.LabelEncoder] = None,
    ) -> None:
        """Construct a new naive Bayes intent classifier using the sklearn framework."""
        super().__init__(component_config)

        if (not clf) + (not le) == 1:
            raise ValueError(
                "Expected classifier and label encoder instance. "
                "Specifying only one of them leads to an incomplete model definition."
            )
        self.le = le
        self.clf = clf

    @abstractmethod
    def create_sklearn_classifier(self, **kwargs: Any) -> sklearn.base.ClassifierMixin:
        r"""Lazily imports the required sklearn classifier class and creates and
        instantiates the sklearn classifier using all the given keyword arguments.

        :param **kwargs: see defaults dictionary
        """
        pass

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Trains the intent classifier from scratch on a data set. A any classifier
        components passed to the constructor will be forgotten, i.e.
        you cannot use it for finetuning yet.

        Reminder: Enabling finetuning here would be possible for sklearn
        models that implement partial_fit or have a warm_start option.

        :param config: unused
        :param **kwargs: unused
        """
        if self.clf is not None:
            rasa.shared.utils.io.raise_warning(
                "Intent classifier has already been trained. Calling train again discards the old training results."
            )

        self.clf = self.create_sklearn_classifier(
            **{key: self.component_config[key] for key in self.defaults}
        )

        X, y, self.le = self._collect_features_and_encode_labels(training_data)

        if len(y):  # i.e. prepared data is not empty
            with warnings.catch_warnings():
                # filter out all sklearn warnigns
                warnings.simplefilter("ignore")
                self.clf.fit(X, y)
        else:
            # if given data is empty, set these components to none explicitly
            # to indicate that nothing was trained
            self.le = None
            self.clf = None

    @staticmethod
    def _collect_features_and_encode_labels(
        training_data: TrainingData,
    ) -> Tuple[scipy.sparse.spmatrix, np.ndarray, sklearn.preprocessing.LabelEncoder]:
        """
        Collects all intent examples from the given training data,
        trains a label encoder and returns this encoder along
        with all sparse features from the intent examples and
        the transformed labels.

        :return: tuple containing the sparse feature matrix of shape
          (n_examples, n_words), an array of shape (n_examples)
          containing the encoded intent labels, and the trained
          label encoder
        """

        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()

        # label (encoder)
        labels = [e.get(INTENT) for e in training_data.intent_examples]
        if len(set(labels)) < 2:
            rasa.shared.utils.io.raise_warning(
                "Can not train an intent classifier as there are not "
                "enough intents. Need at least 2 different intents. "
                "Skipping training of intent classifier.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return (scipy.sparse.coo_matrix([]), np.array([]), le)

        y = le.fit_transform(labels)

        # features
        X = scipy.sparse.vstack(
            [
                SparseSklearnIntentClassifier._get_sentence_features(example)
                for example in training_data.intent_examples
            ]
        )

        return X, y, le

    @staticmethod
    def _get_sentence_features(message: Message) -> scipy.sparse.spmatrix:
        """extracts the sparse sentence features and warns if dense features are present."""
        _, dense_sentence_features = message.get_dense_features(TEXT)
        if dense_sentence_features is not None:
            rasa.shared.utils.io.raise_warning(
                "Dense features are being computed but not used."
            )

        _, sentence_features = message.get_sparse_features(TEXT)
        if sentence_features is None:
            raise ValueError("No sparse sentence features present. Not able to train.")
        return sentence_features.features

    def process(self, message: Message, **kwargs: Any) -> None:
        r"""Computes the most likely intent and its probability for a message and
        adds this information to the message.

        :param **kwargs: unused
        """

        if not self.clf:
            # component is either not trained or didn't
            # receive enough training data
            intent = None
            intent_ranking = []
        else:
            X = self._get_sentence_features(message)
            # note that X is of shape (1, vocabulary_size)
            assert len(X.shape) == 2 and X.shape[0] == 1, "Feature format has changed."
            # hence, sklearn methods work without wrapping the single
            # example into a list
            probabilities = self.predict_proba(X)
            assert LABEL_RANKING_LENGTH > 0, "Unexpected LABEL_RANKING_LENGTH setting."
            intent_indices, intent_probabilities = self.sort_by_scores(
                scores=probabilities, topk=LABEL_RANKING_LENGTH
            )

            # remember that X was of shape (1, vocabulary size)
            # so we need to unravel them here...
            intent_indices = list(intent_indices[0])
            intent_probabilities = list(intent_probabilities[0])

            intents = self.le.inverse_transform(intent_indices)

            intent_ranking = [
                {INTENT_NAME_KEY: intent_name, PREDICTED_CONFIDENCE_KEY: score}
                for intent_name, score in zip(intents, intent_probabilities)
            ]
            intent = {
                INTENT_NAME_KEY: intents[0],
                PREDICTED_CONFIDENCE_KEY: intent_probabilities[0],
            }

        message.set(INTENT, intent, add_to_output=True)
        message.set(INTENT_RANKING_KEY, intent_ranking, add_to_output=True)

    @staticmethod
    def sort_by_scores(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a matrix of scores that defines, for each sample,
        scores for a fixed number of classes, sorts the classes
        according to the respective scores (in decreasing order)
        and returns this sorting along with the scores.
        Moreover, if the topk parameter is specified only the
        results for the top scoring results (per sample) are
        returned.

        :param scores: a matrix of shape (n_samples, n_classes)
        :param topk: limits the number of results to the topk classes
          with the highest scores; disable the limit by setting it to
          a value <=0 or a value >= n_classes
        :return: a tuple
        """
        if len(scores.shape) != 2:
            raise ValueError("Expected 2 dimensional input for score matrix.")
        sorted_indices = np.fliplr(np.argsort(scores, axis=1))
        sorted_scores = np.array([a[b] for a, b in zip(scores, sorted_indices)])
        limit = sorted_scores.shape[1]
        if topk > -1:
            limit = min(topk, limit)
        return sorted_indices[:, :limit], sorted_scores[:, :limit]

    def predict_proba(self, X: scipy.sparse.spmatrix) -> np.ndarray:
        """
        Given bow vectors, predict the probability of each
        possible intent, for each bow vector.

        :param X: bow of input text; matrix of shape (n_samples, n_words)
        :return: probabilities per input text and possible intent; a matrix of shape (n_samples, n_intents)
        """
        return self.clf.predict_proba(X)

    def predict(self, X: scipy.sparse.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given bow vectors, predict the intent label, for each bow vector.

        :param X: bow of input text; matrix of shape (n_examples, n_words)
        :return: intent labels; a matrix of shape (n_examples,)
        """
        return self.clf.predict(X)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        classifier_file_name = file_name + "_classifier.pkl"
        encoder_file_name = file_name + "_encoder.pkl"
        if self.clf and self.le:
            io_utils.json_pickle(
                os.path.join(model_dir, encoder_file_name), self.le.classes_
            )
            io_utils.json_pickle(
                os.path.join(model_dir, classifier_file_name), self.clf
            )
        return {"classifier": classifier_file_name, "encoder": encoder_file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["SparseSklearnIntentClassifier"] = None,
        **kwargs: Any,
    ) -> "SparseSklearnIntentClassifier":
        from sklearn.preprocessing import LabelEncoder

        classifier_file = os.path.join(model_dir, meta.get("classifier"))
        encoder_file = os.path.join(model_dir, meta.get("encoder"))

        if os.path.exists(classifier_file):
            classifier = io_utils.json_unpickle(classifier_file)
            classes = io_utils.json_unpickle(encoder_file)
            encoder = LabelEncoder()
            encoder.classes_ = classes
            return cls(meta, classifier, encoder)
        else:
            return cls(meta)
