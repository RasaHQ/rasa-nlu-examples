# Rasa NLU Examples

<img src="square-logo.svg" width=150 height=150 align="right">

This repository contains some example components meant for educational and inspirational
purposes. These are components that we open source to encourage experimentation but
these are components that are **not officially supported**. There will be some tests
and some documentation but this is a community project, not something that is part of core Rasa.

# Components

The following components are implemented.

## Meta

#### `rasa_nlu_examples.meta.Printer` [docs](docs/meta/printer/)

This component will print what each featurizer adds to the NLU message. Very useful for debugging.

## Dense Featurizers

#### `rasa_nlu_examples.featurizers.dense.FastTextFeaturizer` [docs](docs/featurizer/fasttext/)

These are the pretrained embeddings from FastText, see for more info [here](https://fasttext.cc/).
These are available in 157 languages, see [here](https://fasttext.cc/docs/en/crawl-vectors.html#models).

#### `rasa_nlu_examples.featurizers.dense.BytePairFeaturizer` [docs](docs/featurizer/bytepair/)

These BytePair embeddings are specialized subword embeddings that are built to be lightweight.
See [this link](https://nlp.h-its.org/bpemb/) for more information. These are available in 227 languages and
you can specify the subword vocabulary size as well as the dimensionality.
