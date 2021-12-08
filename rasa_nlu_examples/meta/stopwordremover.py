import typing
import pathlib
from typing import Any, Optional, Text, Dict, List, Type

from flashtext import KeywordProcessor
from rasa.nlu.components import Component
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class StopWordRemover(Component):
    """
    Removes text from the message before it is passed to the tokenizer.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return []

    defaults = {"alias": None, "path": None}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

        if not self.component_config["path"]:
            raise ValueError(
                "You must specify the `path` parameter for StopWordRemover in `config.yml`."
            )

        self.stopwords = (
            pathlib.Path(self.component_config["path"]).read_text().split("\n")
        )
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        for word in self.stopwords:
            self.keyword_processor.add_keyword(word)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        txt = message.get(TEXT)
        found_words = self.keyword_processor.extract_keywords(message.get(TEXT))
        for word in found_words:
            txt = txt.replace(word, "")
        message.set(TEXT, txt.strip())

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

        return cls(meta)
