from typing import Any, Dict, List, Text

import spacy
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


class BlankSpacyTokenizer(Tokenizer):
    """
    A simple spaCy tokenizer without a language model attached.

    This tokenizer implements the tokenizers listed here: https://spacy.io/usage/models#languages
    Note that this tokenizer does not require a SpacyNLP component and that the
    standard SpacyNLP component should be omitted.
    """

    defaults = {
        "lang": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the a blank spaCy model."""
        super().__init__(component_config)
        self.nlp = spacy.blank(component_config["lang"])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
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
