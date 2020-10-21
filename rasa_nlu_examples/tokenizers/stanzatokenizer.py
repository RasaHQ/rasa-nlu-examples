from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
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
        # Where to load the model
        "cache_dir": None,
    }

    # the StanzaTokenizer only supports languages from this list
    supported_language_list = [
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
            lang=component_config["lang"],  # the language model from Stanza to user
            dir=component_config[
                "cache_dir"
            ],  # the caching directory to load the model from
            processors="tokenize,pos,lemma",  # info: https://stanfordnlp.github.io/stanza/pipeline.html#processors
            tokenize_no_ssplit=True,  # disable sentence segmentation
        )

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        doc = self.nlp(text)
        stanza_tokens = reduce(lambda a, b: a + b, doc.sentences).tokens
        # In the code below, if Stanza detects multi-word tokens then we should not
        # fill in the lemma/pos information. Otherwise we're good.
        return [
            Token(
                text=t.text,
                start=t.start_char,
                end=t.end_char,
                lemma=t.words[0].lemma if len(t.words) == 1 else None,
                data={POS_TAG_KEY: t.words[0].pos} if len(t.words) == 1 else None,
            )
            for t in stanza_tokens
        ]
