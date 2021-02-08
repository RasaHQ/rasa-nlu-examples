from rasa_nlu_examples.common import NotInstalled

try:
    from rasa_nlu_examples.tokenizers.stanzatokenizer import StanzaTokenizer
except ImportError:
    StanzaTokenizer = NotInstalled("StanzaTokenizer", "stanza")

try:
    from rasa_nlu_examples.tokenizers.thai_tokenizer import ThaiTokenizer
except ImportError:
    ThaiTokenizer = NotInstalled("ThaiTokenizer", "thai")


__all__ = ["StanzaTokenizer", "ThaiTokenizer"]
