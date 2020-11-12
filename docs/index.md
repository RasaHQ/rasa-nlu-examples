<img src="square-logo.svg" width=40 height=40 style="margin: 10px;" align="right">

# **Rasa NLU Examples**

This repository contains some example components meant for educational and inspirational
purposes. These are components that we open source to encourage experimentation but
these are components that are **not officially supported**. There will be some tests
and some documentation but this is a community project, not something that is part of core Rasa.

The goal of these tools will be to be compatible with the most recent version of
rasa only. You may need to point to an older release of the project if you want
it to be compatible with an older version of Rasa.

The following components are implemented.

## **Tokenizers**

Tokenizers can split up the input text into tokens. Depending on the Tokenizer that you pick
you can also choose to apply lemmatization. For languages that have rich grammatical features
this might help reduce the size of all the possible tokens.

![](images/tokenisation.png)

### StanzaTokenizer

`rasa_nlu_examples.tokenizers.StanzaTokenizer` [docs](docs/tokenizer/thai_tokenizer/)

We support a tokenizier based on [Stanza](https://github.com/stanfordnlp/stanza). This
tokenizer offers part of speech tagging as well as lemmatization for many languages that
spaCy currently does not support. These features might help your ML pipelines in those
situations.

### ThaiTokenizer

`rasa_nlu_examples.tokenizers.ThaiTokenizer` [docs](docs/tokenizer/thai_tokenizer/)

We support a Thai tokenizier based on PyThaiNLP [link](https://github.com/PyThaiNLP/pythainlp).

## **Dense Featurizers**

![](images/dense_features.png)

Dense featurizers attach dense numeric features per token as well as to the entire utterance. These
features are picked up by intent classifiers and entity detectors later in the pipeline.

### FastTextFeaturizer

**`rasa_nlu_examples.featurizers.dense.FastTextFeaturizer` [docs](docs/featurizer/fasttext/)**

These are the pretrained embeddings from FastText, see for more info [here](https://fasttext.cc/).
These are available in 157 languages, see [here](https://fasttext.cc/docs/en/crawl-vectors.html#models).

### BytePairFeaturizer

**`rasa_nlu_examples.featurizers.dense.BytePairFeaturizer` [docs](docs/featurizer/bytepair/)**

These BytePair embeddings are specialized subword embeddings that are built to be lightweight.
See [this link](https://nlp.h-its.org/bpemb/) for more information. These are available in 227 languages and
you can specify the subword vocabulary size as well as the dimensionality.

### GensimFeaturizer

**`rasa_nlu_examples.featurizers.dense.GensimFeaturizer` [docs](docs/featurizer/gensim/)**

A benefit of the `gensim` library is that it is very easy to train your own word embeddings.
It's typically only about 5 lines of code. That means that you could train your own word-embeddings
and then easily use them in a Rasa pipeline. This can be useful if you have reason to believe
that standard training corpora (like Wikipedia) are not valid for your use-case.

## **Fallback Classifiers**

![](images/fallback.png)

Fallback classifiers are models that can override previous intents. In Rasa NLU there is a
[NLU Fallback Classifier](https://rasa.com/docs/rasa/fallback-handoff#nlu-fallback) that
can "fallback" whenever the main classifier isn't confident about their prediction. In this repository
we also host a few of these models such that you can handle specific instances with a custom model too.
These models are meant to be used in combination with a [RulePolicy](https://rasa.com/docs/rasa/policies#rule-based-policies).

### FasttextLanguage

**`rasa_nlu_examples.fallback.FasttextLanguageFallbackClassifier` [docs](docs/fallback/fasttextlanguagefallback.md)**

This fallback classifier is based on [fasttext](https://fasttext.cc/docs/en/language-identification.html). It
can detect when a user is speaking in an unintended language such that you can create a rule to respond
appropriately.

## **Meta**

The components listed here won't effect the NLU pipeline but they might instead cause extra logs
to appear to help with debugging.

### Printer

**`rasa_nlu_examples.meta.Printer` [docs](docs/meta/printer/)**

This component will print what each featurizer adds to the NLU message. Very useful for debugging.
You can find a tutorial on it [here](https://blog.rasa.com/custom-printer-component/).


## Contributing

You can find the contribution guide [here](https://rasahq.github.io/rasa-nlu-examples/contributing/).
