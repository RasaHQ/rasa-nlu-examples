import logging
import pathlib
from typing import Any, Text, Dict, List, Type

import rasa.shared.utils.io
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
from flashtext import KeywordProcessor

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class FlashTextEntityExtractor(EntityExtractorMixin, GraphComponent):
    @classmethod
    def required_components(cls) -> List[Type]:
        return [Tokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["flashtext"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # text will be processed with case insensitive as default
            "case_sensitive": False,
            "non_word_boundaries": "",
            "path": None,
            "entity_name": None,
            "encoding": None,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        name: Text,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.validate_config(config=config)
        config = {**self.get_default_config(), **config}
        self.entity_name = config.get("entity_name")
        self.path = config.get("path")
        self.keyword_processor = KeywordProcessor(
            case_sensitive=config["case_sensitive"]
        )
        for non_word_boundary in config["non_word_boundaries"]:
            self.keyword_processor.add_non_word_boundary(non_word_boundary)
        words = (
            pathlib.Path(self.path).read_text(encoding=config["encoding"]).split("\n")
        )
        if len(words) == 0:
            rasa.shared.utils.io.raise_warning(
                f"No words found in the {pathlib.Path(self.path)} file."
            )
        for word in words:
            self.keyword_processor.add_keyword(word)

    def train(self, training_data: TrainingData) -> Resource:
        pass

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name, model_storage, resource)

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._set_entities(message)
        return messages

    def _set_entities(self, message: Message, **kwargs: Any) -> None:
        matches = self._extract_entities(message)
        message.set(ENTITIES, message.get(ENTITIES, []) + matches, add_to_output=True)

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        if len(self.keyword_processor) == 0:
            return []
        matches = self.keyword_processor.extract_keywords(
            message.get(TEXT), span_info=True
        )
        return [
            {
                ENTITY_ATTRIBUTE_TYPE: self.entity_name,
                ENTITY_ATTRIBUTE_START: match[1],
                ENTITY_ATTRIBUTE_END: match[2],
                ENTITY_ATTRIBUTE_VALUE: match[0],
                "confidence": 1.0,
                "extractor": "FlashTextEntityExtractor",
            }
            for match in matches
        ]

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        if not config.get("entity_name"):
            raise ValueError("FlashTextEntityExtractor requires a `entity_name`.")
        if not config.get("path"):
            raise ValueError("FlashTextEntityExtractor requires a `path`.")
