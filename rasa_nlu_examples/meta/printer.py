import typing
from typing import Any, Optional, Text, Dict, List, Type
import numpy as np

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.constants import TEXT

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class Printer(Component):
    """
    A component that prints the message. Useful for debugging while running `rasa shell`.

    **Configurable Variables:**

    - alias: gives an extra name to the component and adds an extra message that is printed

    Usage:

    ```yml
    language: en

    pipeline:
    - name: WhitespaceTokenizer
    - name: LexicalSyntacticFeaturizer
    - name: rasa_nlu_examples.meta.Printer
      alias: before count vectors
    - name: CountVectorsFeaturizer
      analyzer: char_wb
      min_ngram: 1
      max_ngram: 4
    - name: rasa_nlu_examples.meta.Printer
      alias: after count vectors
    - name: DIETClassifier
      epochs: 1

    policies:
      - name: MemoizationPolicy
      - name: KerasPolicy
      - name: MappingPolicy
    ```

    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return []

    defaults = {"alias": None}
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    @staticmethod
    def _is_list_tokens(v: Any) -> bool:
        if isinstance(v, List):
            if len(v) > 0:
                if isinstance(v[0], Token):
                    return True
        return False

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        if self.component_config["alias"]:
            print("\n")
            print(self.component_config["alias"])
        print(f"text : {message.as_dict()[TEXT]}")
        for k, v in message.data.items():
            if self._is_list_tokens(v):
                print(f"{k}: {[t.text for t in v]}")
            elif isinstance(v, np.ndarray):
                print(f"{k}: Dense array with shape {v.shape}")
            else:
                print(f"{k}: {v.__repr__()}")

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
