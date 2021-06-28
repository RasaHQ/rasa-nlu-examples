from typing import Any

import sklearn
from sklearn.linear_model import LogisticRegression

from rasa_nlu_examples.classifiers.sparse_sklearn_intent_classifier import (
    SparseSklearnIntentClassifier,
)


class SparseLogisticRegressionIntentClassifier(SparseSklearnIntentClassifier):
    r"""A logistic regression classifier using the sklearn framework with sparse features."""

    defaults = {
        # The following parameters and defaults are the same as the ones used by the
        # current scikit-learn version (0.24.2). For some nice explanations on what
        # these parameters and their defaults so, have a look at the scikit-learn docs:
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        "C": 1.0,
        "class_weight": "balanced",
        "dual": False,
        "fit_intercept": True,
        "intercept_scaling": 1,
        "l1_ratio": None,
        "max_iter": 100,
        "multi_class": "auto",
        "n_jobs": None,
        "penalty": "l2",
        "random_state": None,
        "solver": "lbfgs",
        "tol": 0.0001,
        "verbose": 0,
    }

    def create_sklearn_classifier(self, **kwargs: Any) -> sklearn.base.ClassifierMixin:
        r"""Lazily imports the required sklearn classifier class and creates and
        instantiates the sklearn classifier using all the given keyword arguments.

        :param **kwargs: see defaults dictionary
        """

        return LogisticRegression(**kwargs)
