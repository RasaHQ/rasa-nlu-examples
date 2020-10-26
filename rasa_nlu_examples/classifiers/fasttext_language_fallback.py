import os
import logging
import fasttext
from typing import Any, List, Type, Text, Dict, Optional

from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    INTENT_RANKING_KEY,
    TEXT,
    INTENT,
)

from rasa.nlu.constants import TOKENS_NAMES


logger = logging.getLogger(__name__)


class FasttextLanguageFallbackClassifier(IntentClassifier):
    """
    This Classifier uses FastText to detect a language. If a language is *not* detected then it will
    trigger an intent. Such an intent can be used by a Rule Policy to set the correct message.

    More information can be found here:
    https://fasttext.cc/docs/en/language-identification.html
    """

    defaults = {
        "language": None,
        "threshold": 0.7,
        "min_tokens": 3,
        "min_chars": 10,
        "intent_triggered": None,
        "cache_dir": None,
        "file": None,
    }
    language_list = [
        "af",
        "als",
        "am",
        "an",
        "ar",
        "arz",
        "as",
        "ast",
        "av",
        "az",
        "azb",
        "ba",
        "bar",
        "bcl",
        "be",
        "bg",
        "bh",
        "bn",
        "bo",
        "bpy",
        "br",
        "bs",
        "bxr",
        "ca",
        "cbk",
        "ce",
        "ceb",
        "ckb",
        "co",
        "cs",
        "cv",
        "cy",
        "da",
        "de",
        "diq",
        "dsb",
        "dty",
        "dv",
        "el",
        "eml",
        "en",
        "eo",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fr",
        "frr",
        "fy",
        "ga",
        "gd",
        "gl",
        "gn",
        "gom",
        "gu",
        "gv",
        "he",
        "hi",
        "hif",
        "hr",
        "hsb",
        "ht",
        "hu",
        "hy",
        "ia",
        "id",
        "ie",
        "ilo",
        "io",
        "is",
        "it",
        "ja",
        "jbo",
        "jv",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "krc",
        "ku",
        "kv",
        "kw",
        "ky",
        "la",
        "lb",
        "lez",
        "li",
        "lmo",
        "lo",
        "lrc",
        "lt",
        "lv",
        "mai",
        "mg",
        "mhr",
        "min",
        "mk",
        "ml",
        "mn",
        "mr",
        "mrj",
        "ms",
        "mt",
        "mwl",
        "my",
        "myv",
        "mzn",
        "nah",
        "nap",
        "nds",
        "ne",
        "new",
        "nl",
        "nn",
        "no",
        "oc",
        "or",
        "os",
        "pa",
        "pam",
        "pfl",
        "pl",
        "pms",
        "pnb",
        "ps",
        "pt",
        "qu",
        "rm",
        "ro",
        "ru",
        "rue",
        "sa",
        "sah",
        "sc",
        "scn",
        "sco",
        "sd",
        "sh",
        "si",
        "sk",
        "sl",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "tyv",
        "ug",
        "uk",
        "ur",
        "uz",
        "vec",
        "vep",
        "vi",
        "vls",
        "vo",
        "wa",
        "war",
        "wuu",
        "xal",
        "xmf",
        "yi",
        "yo",
        "yue",
        "zh",
    ]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)
        if "cache_dir" not in component_config.keys():
            raise ValueError(
                "You need to set `cache_dir` for the FasttextLanguageFallbackClassifier."
            )
        if "file" not in component_config.keys():
            raise ValueError(
                "You need to set `file` for the FasttextLanguageFallbackClassifier."
            )
        if "intent_triggered" not in component_config.keys():
            raise ValueError(
                "You need to set `intent_triggered` for the FasttextLanguageFallbackClassifier."
            )
        if "language" not in component_config.keys():
            raise ValueError(
                "You need to set `language` for the FasttextLanguageFallbackClassifier."
            )
        if component_config["language"] not in self.language_list:
            raise ValueError(
                "You have specified a unsupported language. See https://fasttext.cc/docs/en/language-identification.html for supported languages."
            )
        path = os.path.join(component_config["cache_dir"], component_config["file"])
        if not os.path.exists(component_config["cache_dir"]):
            raise FileNotFoundError(
                f"It seems that the cache dir {component_config['cache_dir']} does not exists. Please check config."
            )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"It seems that file {path} does not exists. Please check config."
            )

        self.model = fasttext.load_model(path=path)
        self.language = component_config["language"]
        self.min_chars = component_config["min_chars"]
        self.min_tokens = component_config["min_tokens"]
        self.threshold = component_config["threshold"]
        self.intent_triggered = component_config["intent_triggered"]

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [IntentClassifier]

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            message: The :class:`rasa.shared.nlu.training_data.message.Message` to
            process.

        """

        if not self._not_enough_text(message):
            logger.debug(
                "There's not enough text to do proper language identification."
            )
            return

        proba_dict = {
            k: v for k, v in zip(*self.model.predict(message.get(TEXT), k=20))
        }
        proba = proba_dict.get(f"__label__{self.language}", 0.0)
        if proba < self.threshold:
            logger.debug(
                f"FastText thinks this message is not from {self.language} language. Will override and trigger {self.intent_triggered} intent."
            )
            message.data[INTENT] = self.intent_triggered
            message.data.setdefault(INTENT_RANKING_KEY, [])
            message.data[INTENT_RANKING_KEY].insert(0, self.intent_triggered)

    def _not_enough_text(self, message: Message):
        """
        If the message does not have enough text in it then we should not allow our
        language model to run. It's hard to properly estimate a language on a single word.
        """
        n_chars = len(message.get(TEXT).replace(" ", ""))
        n_tokens = len(message.get(TOKENS_NAMES[TEXT]))
        return (n_chars < self.min_chars) | (n_tokens < self.min_tokens)
