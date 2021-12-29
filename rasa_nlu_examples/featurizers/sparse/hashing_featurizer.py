from typing import Any, Optional, Text, Dict, List, Type

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from rasa.engine.storage.resource import Resource
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.featurizers.sparse_featurizer.sparse_featurizer import SparseFeaturizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.nlu.constants import (
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=True
)
class HashingFeaturizer(GraphComponent, SparseFeaturizer):
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
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["sklearn"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        # The following parameters and defaults are similar as the ones used by
        # scikit-learn (version 0.24.2). For detailed explanations on these parameters
        # and their defaults, have a look at the scikit-learn docs:
        # https://scikit-learn.org/0.24/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html

        return {
            **SparseFeaturizer.get_default_config(),
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

    def __init__(self, config: Dict[Text, Any], name: Text) -> None:
        default = self.get_default_config()
        params = {key: config.get(key, default[key]) for key in default}
        config["alias"] = name
        super().__init__(name, config)
        ngram_range = (params.pop("min_ngram"), params.pop("max_ngram"))
        params.pop("alias")

        self.hashing_vectorizer = HashingVectorizer(
            **params,
            token_pattern=r"(?u)\b\w+\b" if params["analyzer"] == "word" else None,
            ngram_range=ngram_range,
            dtype=np.float32,
        )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config, execution_context.node_name)

    def create_word_vector(self, document: List[Text]) -> np.ndarray:
        return self.hashing_vectorizer.fit_transform(document).tocoo()

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the component from training data."""
        pass

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place."""
        self.process(training_data.training_examples)
        return training_data

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
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            text_vector,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def process(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        for message in messages:
            self.set_features(message)
        return messages

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        return None

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        pass
