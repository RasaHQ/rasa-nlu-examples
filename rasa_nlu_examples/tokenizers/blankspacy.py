from __future__ import annotations
from typing import Any, Dict, List, Text

import spacy
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class BlankSpacyTokenizer(Tokenizer):
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
            "lang": None,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        config = {**self.get_default_config(), "lang": config["lang"]}
        super().__init__(config)
        self.nlp = spacy.blank(name=config["lang"])

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> BlankSpacyTokenizer:
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided message."""
        text = message.get(attribute)

        doc = self.nlp(text)
        tokens = [
            Token(
                text=t.text,
                start=t.idx,
            )
            for t in doc
            if t.text and t.text.strip()
        ]
        return self._apply_token_pattern(tokens)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        if "lang" not in config.keys():
            raise ValueError(
                "BlankSpacyTokenizer needs language configured via `lang`."
            )
