Fasttext supports word embeddings for 157 languages and is trained on both Common Crawl and Wikipedia. You can download the embeddings
[here](https://fasttext.cc/docs/en/crawl-vectors.html#models). Note that this featurizer is a *dense* featurizer. Beware that these embedding files tend to be big: about 6-7Gb.

## Configurable Variables

- **cache_dir**: pass it the name of the directory where you've downloaded the embeddings
- **file**: pass it the name of the file that contains the word embeddings

## Base Usage

The configuration file below demonstrates how you might use the fasttext embeddings. In this example
we're building a pipeline for the Dutch language and we're assuming that the embeddings have been
downloaded beforehand and save over at `downloaded/beforehand/cc.nl.300.bin`.

```yaml
language: nl

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.dense.FastTextFeaturizer
  cache_dir: downloaded/beforehand
  file: cc.nl.300.bin
- name: DIETClassifier
  epochs: 100
```
