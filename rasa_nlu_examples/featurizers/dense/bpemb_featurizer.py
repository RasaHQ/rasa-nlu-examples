import typing
from pathlib import Path
from typing import Any, Optional, Text, Dict, List, Type

import numpy as np
from bpemb import BPEmb

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.featurizers.featurizer import DenseFeaturizer

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata
from rasa.nlu.constants import DENSE_FEATURE_NAMES, DENSE_FEATURIZABLE_ATTRIBUTES, TEXT


class BytePairFeaturizer(DenseFeaturizer):
    """This component adds BPEmb features."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["bpemb"]

    defaults = {
        # specifies the language of the subword segmentation model
        "lang": "en",
        # specifies the dimension of the subword embeddings
        "dim": 25,
        # specifies the vocabulary size of the segmentation model
        "vs": 1000,
        # if set to True and the given vocabulary size can't be loaded for the given
        # model, the closest size is chosen
        "vs_fallback": True,
        # specifies the folder in which downloaded BPEmb files will be cached
        "cache_dir": Path.home() / Path(".cache/bpemb"),
        # specifies the path to a custom SentencePiece model file
        "model_file": None,
        # specifies the path to a custom embedding file. Supported formats are Word2Vec
        # plain text and GenSim binary.
        "emb_file": None,
    }

    language_list = [
        "mt",
        "sd",
        "cr",
        "ba",
        "ht",
        "scn",
        "bi",
        "stq",
        "sm",
        "diq",
        "no",
        "yi",
        "vec",
        "bug",
        "am",
        "tl",
        "mn",
        "atj",
        "ko",
        "mai",
        "lij",
        "tcy",
        "sl",
        "bn",
        "dv",
        "rm",
        "ng",
        "ml",
        "kg",
        "koi",
        "war",
        "et",
        "mhr",
        "als",
        "bar",
        "ii",
        "sco",
        "got",
        "pnb",
        "ss",
        "bpy",
        "tum",
        "ru",
        "qu",
        "hy",
        "tw",
        "bm",
        "vep",
        "dty",
        "udm",
        "gd",
        "lbe",
        "rmy",
        "azb",
        "kw",
        "ja",
        "wuu",
        "pag",
        "ro",
        "tet",
        "ee",
        "min",
        "su",
        "ha",
        "glk",
        "pcd",
        "tk",
        "nrm",
        "ku",
        "gn",
        "ty",
        "bh",
        "pap",
        "fr",
        "ia",
        "cs",
        "ky",
        "ff",
        "kab",
        "rn",
        "csb",
        "tt",
        "cy",
        "ilo",
        "kaa",
        "hif",
        "ak",
        "pa",
        "crh",
        "ti",
        "myv",
        "ur",
        "se",
        "uz",
        "cdo",
        "lez",
        "srn",
        "kk",
        "pih",
        "de",
        "an",
        "tyv",
        "ext",
        "gan",
        "wo",
        "si",
        "lmo",
        "hak",
        "az",
        "ka",
        "ik",
        "frr",
        "hsb",
        "ho",
        "af",
        "nds",
        "pam",
        "el",
        "fur",
        "cu",
        "hr",
        "my",
        "nl",
        "da",
        "ch",
        "vls",
        "es",
        "as",
        "lt",
        "ny",
        "so",
        "oc",
        "lad",
        "pnt",
        "ms",
        "bcl",
        "os",
        "co",
        "ks",
        "or",
        "ay",
        "wa",
        "nah",
        "fa",
        "pl",
        "mzn",
        "za",
        "th",
        "fj",
        "kbp",
        "be",
        "zh",
        "ce",
        "sh",
        "sr",
        "id",
        "chy",
        "ps",
        "lo",
        "tr",
        "st",
        "he",
        "ang",
        "sah",
        "io",
        "gom",
        "ki",
        "sn",
        "kbd",
        "jam",
        "bo",
        "pms",
        "sk",
        "kv",
        "ckb",
        "nv",
        "dsb",
        "zea",
        "xmf",
        "fi",
        "ltg",
        "ksh",
        "ve",
        "new",
        "na",
        "jv",
        "tn",
        "sw",
        "rw",
        "ln",
        "bs",
        "gag",
        "ab",
        "olo",
        "is",
        "bjn",
        "ceb",
        "om",
        "vi",
        "ast",
        "uk",
        "mg",
        "mwl",
        "arz",
        "li",
        "mrj",
        "yo",
        "frp",
        "gl",
        "la",
        "km",
        "sv",
        "nap",
        "jbo",
        "bxr",
        "gv",
        "br",
        "fo",
        "ug",
        "pi",
        "bg",
        "ie",
        "din",
        "sa",
        "pdc",
        "cho",
        "lb",
        "ig",
        "aa",
        "sc",
        "fy",
        "kj",
        "eo",
        "eu",
        "kl",
        "sq",
        "to",
        "mi",
        "tpi",
        "kr",
        "hi",
        "arc",
        "ga",
        "nov",
        "mdf",
        "vo",
        "pfl",
        "rue",
        "haw",
        "kn",
        "mh",
        "mr",
        "te",
        "ca",
        "ace",
        "cv",
        "zu",
        "it",
        "iu",
        "av",
        "sg",
        "hz",
        "lv",
        "ts",
        "lrc",
        "ar",
        "hu",
        "nn",
        "nso",
        "krc",
        "mk",
        "tg",
        "ne",
        "dz",
        "ta",
        "mus",
        "ady",
        "en",
        "lg",
        "xal",
        "gu",
        "pt",
        "xh",
        "szl",
        "chr",
    ]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

        self.model = BPEmb(
            lang=self.component_config["lang"],
            dim=self.component_config["dim"],
            vs=self.component_config["vs"],
            vs_fallback=self.component_config["vs_fallback"],
            cache_dir=self.component_config["cache_dir"],
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.set_bpemb_features(example, attribute)

    def create_word_vector(self, document):
        encoded_ids = self.model.encode_ids(document)
        if encoded_ids:
            return self.model.vectors[encoded_ids[0]]

        return np.zeros((self.component_config["dim"],), dtype=np.float32)

    def set_bpemb_features(self, message: Message, attribute: Text = TEXT):
        text_vector = self.create_word_vector(document=message.text)
        word_vectors = [
            self.create_word_vector(document=t.text)
            for t in message.data["tokens"]
            if t.text != "__CLS__"
        ]
        X = np.array(word_vectors + [text_vector])

        features = self._combine_with_existing_dense_features(
            message, additional_features=X, feature_name=DENSE_FEATURE_NAMES[attribute]
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:
        self.set_bpemb_features(message)

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
        if cached_component:
            return cached_component
        else:
            return cls(meta)
