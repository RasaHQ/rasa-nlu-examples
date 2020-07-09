<img src="square-logo.svg" width=40 height=40 style="margin: 10px;" align="right">

# Rasa NLU Examples

This repository contains some example components meant for educational and inspirational
purposes. These are components that we open source to encourage experimentation but
these are components that are **not officially supported**. There will be some tests
and some documentation but this is a community project, not something that is part of core Rasa.

## Components

The goal of these tools will be to be compatible with the most recent version of 
rasa only. You may need to point to an older release of the project if you want
it to be compatible with an older version of Rasa. 

The following components are implemented.

### Meta

#### Printer

**`rasa_nlu_examples.meta.Printer` [docs](docs/meta/printer/)**

This component will print what each featurizer adds to the NLU message. Very useful for debugging.
You can find a tutorial on it [here](https://blog.rasa.com/custom-printer-component/).

### Dense Featurizers

#### FastTextFeaturizer

**`rasa_nlu_examples.featurizers.dense.FastTextFeaturizer` [docs](docs/featurizer/fasttext/)**

These are the pretrained embeddings from FastText, see for more info [here](https://fasttext.cc/).
These are available in 157 languages, see [here](https://fasttext.cc/docs/en/crawl-vectors.html#models).

#### BytePairFeaturizer

**`rasa_nlu_examples.featurizers.dense.BytePairFeaturizer` [docs](docs/featurizer/bytepair/)**

These BytePair embeddings are specialized subword embeddings that are built to be lightweight.
See [this link](https://nlp.h-its.org/bpemb/) for more information. These are available in 227 languages and
you can specify the subword vocabulary size as well as the dimensionality.

## Contributing

You can find the contribution guide [here](https://rasahq.github.io/rasa-nlu-examples/contributing/).
