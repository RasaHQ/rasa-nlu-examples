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

### Tokenizers

#### StanzaTokenizer [docs](docs/tokenizer/stanza/)

We support a tokenizier based on [Stanza](https://github.com/stanfordnlp/stanza). This
tokenizer offers part of speech tagging as well as lemmatization for many languages that
spaCy currently does not support. These features might help your ML pipelines in those
situations.

#### ThaiTokenizer [docs](docs/tokenizer/thai_tokenizer/)

We support a Thai tokenizier based on PyThaiNLP [link](https://github.com/PyThaiNLP/pythainlp).


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

#### GensimFeaturizer

**`rasa_nlu_examples.featurizers.dense.GensimFeaturizer` [docs](docs/featurizer/gensim/)**

A benefit of the `gensim` library is that it is very easy to train your own word embeddings.
It's typically only about 5 lines of code. That means that you could train your own word-embeddings
and then easily use them in a Rasa pipeline. This can be useful if you have reason to believe
that standard training corpora (like Wikipedia) are not valid for your use-case.

## Contributing

You can find the contribution guide [here](https://rasahq.github.io/rasa-nlu-examples/contributing/).
