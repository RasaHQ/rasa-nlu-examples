import pathlib
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message
from functools import reduce
from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY

import stanza


class StanzaTokenizer(Tokenizer):
    """
    The StanzaTokenizer allows for more pos/lemma features to be used in the
    Rasa ML pipelines. It is based on the project found here: https://stanfordnlp.github.io/stanza/usage.html

    Not every language here has good performance metrics. To see the details
    check out this table: https://stanfordnlp.github.io/stanza/performance.html

    Before running the stanza model in production, be sure to check the license information
    since it may differ per language: https://stanfordnlp.github.io/stanza/available_models.html
    """

    defaults = {
        # What language to use
        "lang": None,
        # What package to use
        "pkg": "package",
        # Where to save the model
        "cache_dir": pathlib.Path("~/stanza_resources"),
    }

    # the following language should not be tokenized using the WhitespaceTokenizer
    language_list = [
        "af",
        "grc",
        "ar",
        "hy",
        "eu",
        "be",
        "bg",
        "bxr",
        "ca",
        "zh",
        "lzh",
        "cop",
        "hr",
        "cs",
        "da",
        "nl",
        "en",
        "et",
        "fi",
        "fr",
        "gl",
        "de",
        "got",
        "el",
        "he",
        "hi",
        "hu",
        "id",
        "ga",
        "it",
        "ja",
        "ko",
        "kmr",
        "lv",
        "lt",
        "olo",
        "mt",
        "sme",
        "no",
        "nn",
        "cu",
        "fro",
        "orv",
        "fa",
        "pl",
        "pt",
        "ro",
        "ru",
        "gd",
        "sr",
        "sk",
        "sl",
        "es",
        "sv",
        "swl",
        "ta",
        "te",
        "tr",
        "uk",
        "hsb",
        "ur",
        "ug",
        "vi",
        "wo",
    ]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the Stanza framework."""

        super().__init__(component_config)
        self.nlp = stanza.Pipeline(
            lang=component_config["lang"],
            processors="tokenize,pos,lemma",
            tokenize_no_ssplit=True,
        )

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        doc = self.nlp(text)
        stanza_tokens = reduce(lambda a, b: a + b, doc.sentences).tokens
        tokens = [
            Token(
                text=t.text,
                start=t.start_char,
                end=t.end_char,
                lemma=t.words[0].lemma,
                data={POS_TAG_KEY: t.words[0].pos},
            )
            for t in stanza_tokens
        ]

        return tokens
