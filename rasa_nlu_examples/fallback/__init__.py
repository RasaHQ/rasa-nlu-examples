from rasa_nlu_examples.common import NotInstalled

try:
    from .fasttext_language_fallback import FasttextLanguageFallbackClassifier
except ImportError:
    FasttextLanguageFallbackClassifier = NotInstalled("fasttext", "fasttext")
