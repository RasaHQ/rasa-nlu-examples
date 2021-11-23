from typing import Any, Optional, Text, Dict, List, Type

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.nlu.model import Metadata
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)


class HashingFeaturizer(SparseFeaturizer):
    """This component adds sparse features by applying a hash function to map
    the input text to indices of buckets."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    defaults = {
        # The following parameters and defaults are similar as the ones used by
        # scikit-learn (version 0.24.2). For some nice explanations on these parameters and their defaults,
        # have a look at the scikit-learn docs:
        # https://scikit-learn.org/0.24/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
        "strip_accents": None,
        "lowercase": True,
        "stop_words": None,
        "token_pattern": r"(?u)\b\w+\b",  # do not limit words to >= 2 characters
        "ngram_range": (1, 1),
        "n_features": 2 ** 20,
        "binary": False,
        "norm": "l2",
        "alternate_sign": True,
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        self.hashing_vectorizer = HashingVectorizer(
            **{key: self.component_config[key] for key in self.defaults},
            dtype=np.float32,
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.intent_examples:  # type: ignore
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_features(example, attribute)

    def create_word_vector(self, document: List[Text]) -> np.ndarray:
        return self.hashing_vectorizer.fit_transform(document).tocoo()

    def set_features(self, message: Message, attribute: Text = TEXT) -> None:
        tokens = message.get(TOKENS_NAMES[attribute])

        if not tokens:
            return None

        text_vector = self.create_word_vector(document=[message.get(TEXT)])
        word_vectors = self.create_word_vector(document=[t.text for t in tokens])

        final_sequence_features = Features(
            word_vectors,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.set_features(message)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        return None

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":

        if cached_component:
            return cached_component

        return cls(meta)
