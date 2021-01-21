import logging
import os
import typing
import warnings
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import numpy as np
import scipy.sparse

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

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


class SparseSklearnIntentClassifier(IntentClassifier):
    """Intent classifier using the sklearn framework with sparse features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [SparseFeaturizer]

    defaults = {
        # C parameter of the svm - cross validation will select the best value
        "alpha": [0.1, 0.5, 1.0, 2.0, 10.0],
        # gamma parameter of the svm
        "binarize": [0.0],
        # the kernels to use for the svm training - cross validation will
        # decide which one of them performs best
        "fit_prior": [True],
        # We try to find a good number of cross folds to use during
        # intent training, this specifies the max number of folds
        "max_cross_validation_folds": 5,
        # Scoring function used for evaluating the hyper parameters
        # This can be a name or a function (cfr GridSearchCV doc for more info)
        "scoring_function": "f1_weighted",
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        clf: "sklearn.model_selection.GridSearchCV" = None,
        le: Optional["sklearn.preprocessing.LabelEncoder"] = None,
    ) -> None:
        """Construct a new intent classifier using the sklearn framework."""
        from sklearn.preprocessing import LabelEncoder

        super().__init__(component_config)

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()
        self.clf = clf

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    def transform_labels_str2num(self, labels: List[Text]) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.
        :param labels: List of labels to convert to numeric representation"""

        return self.le.fit_transform(labels)

    def transform_labels_num2str(self, y: np.ndarray) -> np.ndarray:
        """Transforms a list of strings into numeric label representation.
        :param y: List of labels to convert to numeric representation"""

        return self.le.inverse_transform(y)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train the intent classifier on a data set."""

        num_threads = kwargs.get("num_threads", 1)

        labels = [e.get("intent") for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            rasa.shared.utils.io.raise_warning(
                "Can not train an intent classifier as there are not "
                "enough intents. Need at least 2 different intents. "
                "Skipping training of intent classifier.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        y = self.transform_labels_str2num(labels)
        X = scipy.sparse.vstack(
            [
                self._get_sentence_features(example)
                for example in training_data.intent_examples
            ]
        )

        self.clf = self._create_classifier(num_threads, y)

        with warnings.catch_warnings():
            # sklearn raises lots of
            # "UndefinedMetricWarning: F - score is ill - defined"
            # if there are few intent examples, this is needed to prevent it
            warnings.simplefilter("ignore")
            self.clf.fit(X, y)

    @staticmethod
    def _get_sentence_features(message: Message) -> np.ndarray:
        _, sentence_features = message.get_sparse_features(TEXT)
        if sentence_features is not None:
            return sentence_features.features

        raise ValueError(
            "No sentence features present. Not able to train sklearn policy."
        )

    def _num_cv_splits(self, y: np.ndarray) -> int:
        folds = self.component_config["max_cross_validation_folds"]
        return max(2, min(folds, np.min(np.bincount(y)) // 5))

    def _create_classifier(
        self, num_threads: int, y: np.ndarray
    ) -> "sklearn.model_selection.GridSearchCV":
        from sklearn.model_selection import GridSearchCV
        from sklearn.naive_bayes import BernoulliNB

        alpha = self.component_config["alpha"]
        binarize = self.component_config["binarize"]
        fit_prior = self.component_config["fit_prior"]
        # dirty str fix because sklearn is expecting
        # str not instance of basestr...
        tuned_parameters = [
            {"alpha": alpha, "binarize": binarize, "fit_prior": fit_prior}
        ]

        # aim for 5 examples in each fold

        cv_splits = self._num_cv_splits(y)

        return GridSearchCV(
            BernoulliNB(alpha=1, binarize=0.0, fit_prior=True),
            param_grid=tuned_parameters,
            n_jobs=num_threads,
            cv=cv_splits,
            scoring=self.component_config["scoring_function"],
            verbose=1,
            iid=False,
        )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely intent and its probability for a message."""

        if not self.clf:
            # component is either not trained or didn't
            # receive enough training data
            intent = None
            intent_ranking = []
        else:
            X = self._get_sentence_features(message)  #.reshape(1, -1)
            intent_ids, probabilities = self.predict(X)
            intents = self.transform_labels_num2str(np.ravel(intent_ids))
            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten
            probabilities = probabilities.flatten()
            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents), list(probabilities)))[
                    :LABEL_RANKING_LENGTH
                ]

                intent = {"name": intents[0], "confidence": probabilities[0]}

                intent_ranking = [
                    {"name": intent_name, "confidence": score}
                    for intent_name, score in ranking
                ]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """Given a bow vector of an input text, predict the intent label.
        Return probabilities for all labels.
        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given a bow vector of an input text, predict most probable label.
        Return only the most likely label.
        :param X: bow of input text
        :return: tuple of first, the most probable label and second,
                 its probability."""

        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of
        # the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        classifier_file_name = file_name + "_classifier.pkl"
        encoder_file_name = file_name + "_encoder.pkl"
        if self.clf and self.le:
            io_utils.json_pickle(
                os.path.join(model_dir, encoder_file_name), self.le.classes_
            )
            io_utils.json_pickle(
                os.path.join(model_dir, classifier_file_name), self.clf.best_estimator_
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
