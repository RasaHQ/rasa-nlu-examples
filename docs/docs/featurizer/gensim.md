This pages discusses some properties of the `GensimFeaturizer`.
Note that this featurizer is a *dense* featurizer.

Gensim is a popular python library that makes it relatively easy to train your own word vectors. This can be useful if your corpus is very different than what most popular embeddings are trained on. We'll give a small guide on how to train your own embeddings here but you can also read the guide on the [gensim docs](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#training-your-own-model).

## Training Your Own

Training your own gensim model doesn't require a lot of code.

```python
from gensim.models import Word2Vec

text = ["this is a sentence", "so is this", "and we're all talking"]
tokens = [t.split(" ") for t in text]

model = Word2Vec(sentences=tokens, size=10, window=3,
                 min_count=1, iter=5, workers=2)

model.wv.save("wordvectors.kv")
```

This `wordvectors.kv` file should contain all the vectors that you've trained. It's this
file that you can pass on to this component.

## Configurable Variables

- **cache_dir**: pass it the name of the directory where you've downloaded/saved the embeddings
- **file**: pass it the name of the `.kv` file that contains the word embeddings

## Base Usage

The configuration file below demonstrates how you might use the fasttext embeddings. In this example
we're building a pipeline for the an English language and we're assuming that you've trained your
own embeddings which have been saved upfront as `saved/beforehand/filename.kv`.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.dense.GensimFeaturizer
  cache_dir: saved/beforehand
  file: filename.kv
- name: DIETClassifier
  epochs: 100

policies:
  - name: MemoizationPolicy
  - name: KerasPolicy
  - name: MappingPolicy
```
