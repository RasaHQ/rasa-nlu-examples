from rasa_nlu_examples.common import NotInstalled

try:
    from .fasttext_featurizer import FastTextFeaturizer
except ImportError:
    FastTextFeaturizer = NotInstalled("fasttext", "fasttext")

from .bpemb_featurizer import BytePairFeaturizer

from .gensim_featurizer import GensimFeaturizer


__all__ = ["BytePairFeaturizer", "FastTextFeaturizer", "GensimFeaturizer"]
