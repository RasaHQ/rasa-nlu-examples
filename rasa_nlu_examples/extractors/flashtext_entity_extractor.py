import os
from typing import List, Type, Optional, Dict, Text, Any

import rasa.shared.utils.io

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import (
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITIES,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from flashtext import KeywordProcessor


class FlashTextEntityExtractor(EntityExtractor):

    defaults = {
        # text will be processed with case insensitive as default
        "case_sensitive": False,
        "non_word_boundaries": "",
    }

    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        lookups: Optional[Dict[Text, List[Text]]] = None,
    ):
        """This component extracts entities using lookup tables."""

        super().__init__(component_config)
        self.keyword_processor = KeywordProcessor(
            case_sensitive=self.component_config["case_sensitive"]
        )
        for non_word_boundary in self.component_config["non_word_boundaries"]:
            self.keyword_processor.add_non_word_boundary(non_word_boundary)
        if lookups:
            self.keyword_processor.add_keywords_from_dict(lookups)
            self.lookups = lookups

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        lookups = self._extract_lookups(
            training_data,
            use_only_entities=True,
        )

        if len(lookups.keys()) == 0:
            rasa.shared.utils.io.raise_warning(
                "No lookup tables defined in the training data that have a "
                "name equal to any entity in the training data. In order for "
                "this component to work you need to define valid lookup tables "
                "in the training data."
            )
        self.lookups = lookups
        self.keyword_processor.add_keywords_from_dict(lookups)

    def process(self, message: Message, **kwargs: Any) -> None:
        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )

    def _extract_lookups(
        self, training_data: TrainingData, use_only_entities: True
    ) -> Dict[Text, List[Text]]:
        if not training_data.lookup_tables or len(training_data.lookup_tables) == 0:
            return {}
        return {
            lookup_table["name"]: lookup_table["elements"]
            for lookup_table in training_data.lookup_tables
            if (
                not use_only_entities
                or (
                    use_only_entities and lookup_table["name"] in training_data.entities
                )
            )
        }

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        if len(self.keyword_processor) == 0:
            return []
        matches = self.keyword_processor.extract_keywords(
            message.get(TEXT), span_info=True
        )

        return [
            {
                ENTITY_ATTRIBUTE_TYPE: match[0],
                ENTITY_ATTRIBUTE_START: match[1],
                ENTITY_ATTRIBUTE_END: match[2],
                ENTITY_ATTRIBUTE_VALUE: message.get(TEXT)[match[1] : match[2]],
                "confidence": 1.0,
            }
            for match in matches
        ]

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["FlashTextEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "FlashTextEntityExtractor":

        file_name = meta.get("file")
        lookup_file = os.path.join(model_dir, file_name)

        if os.path.exists(lookup_file):
            lookups = rasa.shared.utils.io.read_json_file(lookup_file)
            return FlashTextEntityExtractor(meta, lookups)

        return FlashTextEntityExtractor(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = f"{file_name}.json"
        lookup_file = os.path.join(model_dir, file_name)
        rasa.shared.utils.io.dump_obj_as_json_to_file(lookup_file, self.lookups)
        return {"file": file_name}
