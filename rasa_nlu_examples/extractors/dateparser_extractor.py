import re
from typing import List, Type, Optional, Dict, Text, Any

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
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

from dateparser.search import search_dates


class DateparserEntityExtractor(EntityExtractor):

    defaults = {
        # text will be processed with case insensitive as default
        "prefer_dates_from": None,
        "languages": None,
    }

    def required_components(cls) -> List[Type[Component]]:
        return []

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        lookups: Optional[Dict[Text, List[Text]]] = None,
    ):
        """This component extracts entities using lookup tables."""

        super().__init__(component_config)
        self.settings = {}
        # Dateparser is picky about the dictionary it receives, when value = None,
        # there should be no entry in the dictionary.
        if component_config.get("prefer_dates_from"):
            self.settings["PREFER_DATES_FROM"] = component_config.get(
                "prefer_dates_from"
            )
        self.languages = component_config.get("languages")

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        hits = search_dates(
            message.get(TEXT),
            languages=self.languages if self.languages else None,
            settings=self.settings,
        )
        if not hits:
            return []

        matches = []
        for substr, timestamp in hits:
            for match in re.finditer(substr, message.get(TEXT)):
                matches.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: "DATETIME_REFERENCE",
                        ENTITY_ATTRIBUTE_START: match.start(),
                        ENTITY_ATTRIBUTE_END: match.end(),
                        ENTITY_ATTRIBUTE_VALUE: message.get(TEXT)[
                            match.start() : match.end()
                        ],
                        "parsed_date": str(timestamp),
                        "confidence": 1.0,
                    }
                )
        return matches

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["DateparserEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "DateparserEntityExtractor":
        if cached_component:
            return cached_component

        return cls(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        pass
