import typing
import scipy.sparse
import numpy as np
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.tokenizers.tokenizer import Tokenizer

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata
from rasa.nlu.constants import SPARSE_FEATURE_NAMES, TEXT

from langdetect import detect
from langdetect import DetectorFactory

DetectorFactory.seed = 0


class FastTextFeaturizer(SparseFeaturizer):
    """This component adds fasttext features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["langdetect"]

    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        self.all_langs = {}

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        # We need to know all the languages in the training data
        detected_langs = [detect(example.text) for example in training_data.intent_examples]
        self.all_langs = {lang: i for i, lang in enumerate(set(detected_langs))}

        # If only one language is detected this feature is meaningless
        if len(self.all_langs) == 1:
            raise RuntimeError("We've only been able to detect one language in training data!")

        # Add all the features
        for example in training_data.intent_examples:
            self.set_lang_features(example, attribute)

    def feature_vec(self, text):
        """Give it text and it will output correct numpy array"""
        key = detect(text)
        result = np.zeros(len(self.all_langs)).astype(np.int)
        result[self.all_langs[key]] = 1

        return scipy.sparse.coo_matrix(result)

    def set_lang_features(self, message: Message, attribute: Text = TEXT):
        text_vector = self.feature_vec(message.text)
        word_vectors = [
            self.feature_vec(t.text) for t in message.data["tokens"] if t.text != "__CLS__"
        ]
        X = np.array(word_vectors + [text_vector])  # remember, we need one for __CLS__

        features = self._combine_with_existing_sparse_features(
            message, additional_features=X, feature_name=SPARSE_FEATURE_NAMES[attribute]
        )
        message.set(SPARSE_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.set_lang_features(message)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass

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
        else:
            return cls(meta)
