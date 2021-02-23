# Rasa NLU Examples

<img src="square-logo.svg" width=200 height=200 align="right">

This repository contains Rasa compatible machine learning components. These components
are open sourced in order to encourage experimentation and to quickly offer support to
more tools. By hosting these components here they do not need to go through the same
vetting process as the components in Rasa and we hope that this makes it easier for
people to contribute new ideas.

The components in the repository are **not officially supported**. There will be units tests
as well as documentation but this project should be considered a community project,
not something that is part of core Rasa. If there's a component here that turns out to be
useful to the larger Rasa community then we might port features from this repository to Rasa.

# Install

To use these tools locally you need to install via git.

```python
python -m pip install "rasa_nlu_examples @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
```

Note that if you want to install optional dependencies as well that you'll need to run:

```python
python -m pip install "rasa_nlu_examples[stanza] @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
python -m pip install "rasa_nlu_examples[thai] @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
python -m pip install "rasa_nlu_examples[fasttext] @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
python -m pip install "rasa_nlu_examples[all] @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
```

# Contribute

There are many ways you can contribute to this project.

- You can suggest new features.
- You can help review new features.
- You can submit new components.
- You can let us know if there are bugs.
- You can share the results of an experiment you ran using these tools.
- You can let us know if the components in this library help you.

Feel free to start the discussion by opening an issue on this repository. Before submitting code
to the repository it would help if you first create an issue so that the maintainers can disucss
the changes you would like to contribute. A more in-depth contribution guide can be found
[here](https://rasahq.github.io/rasa-nlu-examples/contributing/).

To get started locally you can run:

```
python -m pip install -e ".[dev]"
pre-commit install
python tests/prepare_everything.py
```

Alternatively you may also run this via the `Makefile`:

```
make install
```

# Documentation

You can find the documentation for this project [here](https://rasahq.github.io/rasa-nlu-examples/).

# Compatibility

This project currently supports components for Rasa 2.0. For older versions, see the list below.

- [version 0.1.3](https://github.com/RasaHQ/rasa-nlu-examples/tree/0.1.3) is the final release for Rasa 1.10

# Features

The following components are implemented;

## **Tokenizers**

Tokenizers can split up the input text into tokens. Depending on the Tokenizer that you pick
you can also choose to apply lemmatization. For languages that have rich grammatical features
this might help reduce the size of all the possible tokens.

![](docs/images/tokenisation.png)

### StanzaTokenizer

`rasa_nlu_examples.tokenizers.StanzaTokenizer` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/tokenizer/thai_tokenizer/)

We support a tokenizier based on [Stanza](https://github.com/stanfordnlp/stanza). This
tokenizer offers part of speech tagging as well as lemmatization for many languages that
spaCy currently does not support. These features might help your ML pipelines in those
situations.

### ThaiTokenizer

`rasa_nlu_examples.tokenizers.ThaiTokenizer` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/tokenizer/thai_tokenizer/)

We support a Thai tokenizier based on PyThaiNLP [link](https://github.com/PyThaiNLP/pythainlp).

## **Dense Featurizers**

![](docs/images/dense_features.png)

Dense featurizers attach dense numeric features per token as well as to the entire utterance. These
features are picked up by intent classifiers and entity detectors later in the pipeline.

### FastTextFeaturizer

**`rasa_nlu_examples.featurizers.dense.FastTextFeaturizer` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/featurizer/fasttext/)**

These are the pretrained embeddings from FastText, see for more info [here](https://fasttext.cc/).
These are available in 157 languages, see [here](https://fasttext.cc/docs/en/crawl-vectors.html#models).

### BytePairFeaturizer

**`rasa_nlu_examples.featurizers.dense.BytePairFeaturizer` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/featurizer/bytepair/)**

These BytePair embeddings are specialized subword embeddings that are built to be lightweight.
See [this link](https://nlp.h-its.org/bpemb/) for more information. These are available in 227 languages and
you can specify the subword vocabulary size as well as the dimensionality.

### GensimFeaturizer

**`rasa_nlu_examples.featurizers.dense.GensimFeaturizer` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/featurizer/gensim/)**

A benefit of the [gensim](https://radimrehurek.com/gensim/) library is that it is very easy to
train your own word embeddings. It's typically only about 5 lines of code. That means that you
could train your own word-embeddings and then easily use them in a Rasa pipeline. This can be
useful if you have specific jargon you'd like to capture.

Another benefit of the tool is that it has made it easy for community members to train custom
embeddings for many languages. Here's a list of resources;

- [AraVec](https://github.com/bakrianoo/aravec#download) has embeddings for Arabic trained on twitter and/or Wikipedia.

## **Sparse Featurizers**

### SemanticMapFeaturizer

**`rasa_nlu_examples.featurizers.sparse.SemanticMapFeaturizer` [docs](docs/featurizer/semantic_map/)**

The `SemanticMapFeaturizer` is an experimental *sparse* featurizer developed by Rasa.
It can only be used in combination with pre-trained embedding files, which you can
find [here](https://github.com/RasaHQ/rasa-embeddings/tree/main/embeddings/semantic_map).
Please refer to our [blog posts](https://blog.rasa.com/exploring-semantic-map-embeddings-1/) for more details.

## **Fallback Classifiers**

![](docs/images/fallback.png)

Fallback classifiers are models that can override previous intents. In Rasa NLU there is a
[NLU Fallback Classifier](https://rasa.com/docs/rasa/fallback-handoff#nlu-fallback) that
can "fallback" whenever the main classifier isn't confident about their prediction. In this repository
we also host a few of these models such that you can handle specific instances with a custom model too.
These models are meant to be used in combination with a [RulePolicy](https://rasa.com/docs/rasa/policies#rule-based-policies).

### FasttextLanguage

**`rasa_nlu_examples.fallback.FasttextLanguageFallbackClassifier` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/fallback/fasttextlanguagefallback.md)**

This fallback classifier is based on [fasttext](https://fasttext.cc/docs/en/language-identification.html). It
can detect when a user is speaking in an unintended language such that you can create a rule to respond
appropriately.

## **Meta**

The components listed here won't effect the NLU pipeline but they might instead cause extra logs
to appear to help with debugging.

### Printer

**`rasa_nlu_examples.meta.Printer` [docs](https://rasahq.github.io/rasa-nlu-examples/docs/meta/printer/)**

This component will print what each featurizer adds to the NLU message. Very useful for debugging.

## **Entity Extraction**

### Name Lists

Language models in spaCy are typically trained on Western news datasets. That means
that the reported benchmarks might not apply to your use-case. For example; detecting
names in texts from France is not the same thing as detecting names in Madagascar. Even
thought French is used actively in both countries, the names of it's citizens might
be so different that you cannot assume that the benchmarks apply universally.

To remedy this we've started collecting name lists. These can be used as a lookup table
which can be picked up  by Rasa's [RegexEntityExtractor](https://rasa.com/docs/rasa/components#regexentityextractor). It
won't be 100% perfect but it should give a reasonable starting point.

You can find the namelists [here](https://github.com/RasaHQ/rasa-nlu-examples/tree/master/data/namelists).
We currently offer namelists for the United States, Germany as well as common Arabic names.
Feel free to submit PRs for more languages. We're also eager to receive feedback.

# Usage

You can install the examples from this repo via pip;

```
pip install git+https://github.com/RasaHQ/rasa-nlu-examples
```

Once installed you can add tools to your `config.yml` file, here's an example;

```yaml
language: en
pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
  OOV_token: oov.txt
  analyzer: word
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.dense.BytePairFeaturizer
  lang: en
  vs: 1000
  dim: 25
- name: DIETClassifier
  epochs: 200
```

An example config for using the Thai tokenizer would look like:

```yaml
language: th
pipeline:
  - name: rasa_nlu_examples.tokenizers.ThaiTokenizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 200
```

And you can use this file to run benchmarks. From the root folder of the project typically
that means running something like;

```
rasa test nlu --config basic-bytepair-config.yml \
          --cross-validation --runs 1 --folds 2 \
          --out gridresults/basic-bytepair-config
```

# Open an Issue

If you've spotted a bug then you can submit an issue [here](https://github.com/RasaHQ/rasa-nlu-examples/issues).
GitHub issues allow us to keep track of a conversation about this repository and it is the preferred
communication channel for bugs related to this project.
