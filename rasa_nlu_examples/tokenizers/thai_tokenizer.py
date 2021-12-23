from __future__ import annotations
from typing import Any, Dict, List, Text

from pythainlp import word_tokenize
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class ThaiTokenizer(Tokenizer):
    """Creates features for entity extraction."""

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["spacy"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # This *must* be added due to the parent class.
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",
            # This is the spaCy language setting.
            "case_sensitive": True,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        config = {**self.get_default_config(), **config}
        super().__init__(config)
        self.case_sensitive = config["case_sensitive"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> ThaiTokenizer:
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided message."""
        text = message.get(attribute)

        if not self.case_sensitive:
            text = text.lower()
        words = word_tokenize(text)

        if not words:
            words = [text]

        return self._convert_words_to_tokens(words, text)
