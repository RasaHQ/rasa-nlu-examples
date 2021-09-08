import os
from typing import Any, Optional, Text, Dict, List, Type

import pyphen
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import rasa.utils.io as io_utils
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


class HyphenFeaturizer(SparseFeaturizer):
    """This component adds sparse BPEmb features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["pyphen"]

    defaults = {}

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        self.dic = pyphen.Pyphen(lang="en_GB")

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        texts = [
            " ".join(self.dic.inserted(e.get("text")).split("-", -1))
            for e in training_data.intent_examples
        ]
        self.countvectorizer = CountVectorizer().fit(texts)

        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_features(example, attribute)

    def create_word_vector(self, document: List[Text]) -> np.ndarray:
        texts = [
            " ".join(self.dic.inserted(e).split("-", -1)) for e in document
        ]
        return self.countvectorizer.transform(texts).tocoo()

    def set_features(self, message: Message, attribute: Text = TEXT) -> None:
        tokens = message.get(TOKENS_NAMES[attribute])

        if not tokens:
            return None

        # We need to reshape here such that the shape is equivalent to that of sparsely
        # generated features. Without it, it'd be a 1D tensor. We need 2D (n_utterance, n_dim).
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
        file_name = file_name + ".pkl"
        vocab = self.countvectorizer.vocabulary_
        featurizer_file = os.path.join(model_dir, file_name)
        io_utils.json_pickle(featurizer_file, vocab)
        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        file_name = meta.get("file")
        featurizer_file = os.path.join(model_dir, file_name)
        vocabulary = io_utils.json_unpickle(featurizer_file)
        new_feat = cls(meta)
        new_feat.countvectorizer = CountVectorizer(vocabulary=vocabulary)
        return new_feat
