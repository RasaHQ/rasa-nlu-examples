import os
import typing
from functools import reduce
from typing import Any, Optional, Text, Dict, List, Type

from gensim.models import KeyedVectors

import numpy as np
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class GensimFeaturizer(DenseFeaturizer):
    """This component adds gensim features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["gensim"]

    defaults = {"file": None, "cache_dir": None}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        if "cache_dir" not in component_config.keys():
            raise ValueError("You need to set `cache_dir` for the GensimFeaturizer.")
        if "file" not in component_config.keys():
            raise ValueError("You need to set `file` for the GensimFeaturizer.")
        path = os.path.join(component_config["cache_dir"], component_config["file"])

        if not os.path.exists(component_config["cache_dir"]):
            raise FileNotFoundError(
                f"It seems that the cache dir {component_config['cache_dir']} does not exists. Please check config."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"It seems that file {path} does not exists. Please check config."
            )

        self.kv = KeyedVectors.load(path)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.training_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_gensim_features(example, attribute)

    def set_gensim_features(self, message: Message, attribute: Text = TEXT) -> None:
        tokens = message.get(TOKENS_NAMES[attribute])

        if not tokens:
            return None

        # If the key is not available then we featurize it with an array of zeros
        word_vectors = np.array(
            [
                self.kv[t.text] if t.text in self.kv else np.zeros(self.kv.vector_size)
                for t in tokens
            ]
        )

        # Sum up all the word vectors so that we have one for __CLS__
        text_vector = reduce(lambda a, b: a + b, word_vectors).reshape(1, -1)

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
        self.set_gensim_features(message)

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
        """Load this component from file."""

        if cached_component:
            return cached_component

        return cls(meta)
