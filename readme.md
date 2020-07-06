<img src="square-logo.svg" width=200 height=200 align="right">

This repository contains some example components meant for educational/inspirational
purposes. These are components that we open source to encourage experimentation but
these are components that are **not officially supported**. There will be some tests
as well as some documentation but this project should be considered a community project,
not something that is part of core Rasa.

# Documentation

You can find the documentation for this project [here](https://rasahq.github.io/rasa-nlu-examples/).
# Components

The following components are implemented.

### Meta

- `rasa_nlu_examples.meta.Printer`: a printer that's useful for debugging

### Dense Featurizers

- `rasa_nlu_examples.featurizers.dense.FastTextFeaturizer`: pretrained fasttext embeddings [link](https://fasttext.cc/)
- `rasa_nlu_examples.featurizers.dense.BytePairFeaturizer`: pretrained byte-pair embeddings [link](https://nlp.h-its.org/bpemb/)

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

And you can use this file to run benchmarks. From the root folder of the project typically
that means running something like;

```
rasa test nlu --config basic-bytepair-config.yml \
          --cross-validation --runs 1 --folds 2 \
          --out gridresults/basic-bytepair-config
```
