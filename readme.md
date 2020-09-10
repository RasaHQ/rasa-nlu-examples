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

# Contribute 

There are many ways you can contribute to this project. 

- You can suggest new features.
- You can help review new features. 
- You can submit new components.
- You can let us know if there are bugs.
- You can let us know if the components in this library help you.

Feel free to start the discussion by opening an issue on this repository. Before submitting code
to the repository it would help if you first create an issue so that the maintainers can disucss
the changes you would like to contribute. A more in-depth contribution guide can be found 
[here](https://rasahq.github.io/rasa-nlu-examples/contributing/).

# Compatibility

The goal is to keep this project compatible with the most recent stable Rasa release. You can find
older versions in the github releases.

Currently, this project is compatible with Rasa 1.10. 

# Documentation

You can find the documentation for this project [here](https://rasahq.github.io/rasa-nlu-examples/).

# Features

The following components are implemented;

### Meta

- `rasa_nlu_examples.meta.Printer`: a printer that's useful for debugging

### Dense Featurizers

- `rasa_nlu_examples.featurizers.dense.GensimFeaturizer`: custom [Gensim](https://radimrehurek.com/gensim/) embeddings 
- `rasa_nlu_examples.featurizers.dense.FastTextFeaturizer`: [FastText](https://fasttext.cc/) embeddings for 157 languages
- `rasa_nlu_examples.featurizers.dense.BytePairFeaturizer`: [BytePair](https://nlp.h-its.org/bpemb/) embeddings for 275 languages

### Tokenizers

- `rasa_nlu_examples.tokenizers.StanzaTokenizer`: a tokenizer that adds lemma/pos features based on [Stanza](https://stanfordnlp.github.io/stanza/) for 63 languages
- `rasa_nlu_examples.tokenizers.ThaiTokenizer`: a Thai tokenizer based on [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)

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
  token_pattern: (?u)\b\w+\b
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
