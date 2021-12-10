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
    """
    This component adds sparse features by applying a hashing vectorizer to the input tokens. Each token gets mapped
    to an index in a sparse vector that is computed by applying a MurmurHash3 function and wrapping the resulting
    integer to the number of columns in the vector.

    The component is stateless and does not require training. For a small number of features, hash collisions get
    more likely, meaning that multiple words get mapped to the same index. It is not possible to compute the inverse
    transform to determine which tokens belong to a specific vector representation.

    The implementation used is Scikit-Learn's HashingVectorizer class, which is described in detail here:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["sklearn"]

    defaults = {
        # The following parameters and defaults are similar as the ones used by
        # scikit-learn (version 0.24.2). For detailed explanations on these parameters
        # and their defaults, have a look at the scikit-learn docs:
        # https://scikit-learn.org/0.24/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
        #
        # the number of columns in the output vector
        "n_features": 2048,  # int
        # whether to use word or character n-grams
        # 'char_wb' creates character n-grams inside word boundaries
        # n-grams at the edges of words are padded with space.
        "analyzer": "word",  # use 'char' or 'char_wb' for character
        # remove accents during the preprocessing step
        "strip_accents": None,  # {'ascii', 'unicode', None}
        # list of stop words
        "stop_words": None,  # string {'english'}, list, or None (default)
        # if `True` convert all characters to lowercase
        "lowercase": True,  # bool
        # set range of ngrams to be extracted
        "min_ngram": 1,
        "max_ngram": 1,
        # if `True`, all non zero counts are set to 1.
        "binary": False,  # bool
        # the norm used to normalize term vectors
        "norm": "l2",  # 'l1', 'l2', or None
        # when `True`, an alternating sign is added to the features as to approximately
        # conserve the inner product in the hashed space even for small n_features.
        "alternate_sign": True,  # bool
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

        params = {key: self.component_config[key] for key in self.defaults}
        ngram_range = (params.pop("min_ngram"), params.pop("max_ngram"))

        self.hashing_vectorizer = HashingVectorizer(
            **params,
            token_pattern=r"(?u)\b\w+\b" if params["analyzer"] == "word" else None,
            ngram_range=ngram_range,
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
