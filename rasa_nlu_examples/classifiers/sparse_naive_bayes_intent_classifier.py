from typing import Any

import sklearn
from sklearn.naive_bayes import BernoulliNB

from rasa_nlu_examples.classifiers.sparse_sklearn_intent_classifier import (
    SparseSklearnIntentClassifier,
)


class SparseNaiveBayesIntentClassifier(SparseSklearnIntentClassifier):
    r"""A naive Bayes intent classifier using the sklearn framework with sparse features."""

    defaults = {
        # Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
        "alpha": 1.0,
        # Threshold for binarizing (mapping to booleans) of sample features.
        # If None, input is presumed to already consist of binary vectors.
        "binarize": 0.0,
        # Whether to learn class prior probabilities or not.
        # If false, a uniform prior will be used.
        "fit_prior": True,
        # Prior probabilities of the classes.
        # If specified the priors are not adjusted according to the data.
        "class_prior": None,
    }

    def create_sklearn_classifier(self, **kwargs: Any) -> sklearn.base.ClassifierMixin:
        r"""Lazily imports the required sklearn classifier class and creates and
        instantiates the sklearn classifier using all the given keyword arguments.

        :param **kwargs: see defaults dictionary
        """
        return BernoulliNB(**kwargs)
