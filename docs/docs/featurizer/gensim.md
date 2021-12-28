This page discusses some properties of the `GensimFeaturizer`.
Note that this featurizer is a *dense* featurizer.

Gensim is a popular python library that makes it relatively easy to
train your own word vectors. This can be useful if your corpus is very
different than what most popular embeddings are trained on. We'll give
a small guide on how to train your own embeddings here but you can
also read the guide on the
[gensim docs](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#training-your-own-model).

## Training Your Own

Training your own gensim model can be done in a few lines of code. A demonstration is shown below.

```python
from gensim.models import Word2Vec

# Gensim needs a list of lists to represent tokens in a document.
# In real life you’d read a text file and turn it into lists here.
text = ["this is a sentence", "so is this", "and we're all talking"]
tokens = [t.split(" ") for t in text]

# This is where we train new word embeddings.
model = Word2Vec(sentences=tokens, size=10, window=3,
                 min_count=1, iter=5, workers=2)

# This is where they are saved to disk.
model.wv.save("wordvectors.kv")
```

This `wordvectors.kv` file should contain all the vectors that you've trained. It's this
file that you can pass on to this component.

## Configurable Variables

- **cache_path**: pass it the name of the filepath where you've downloaded/saved the embeddings

## Base Usage

The configuration file below demonstrates how you might use the gensim embeddings. In this example
we're building a pipeline for the English language and we're assuming that you've trained your
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
  cache_path: path/to/filename.kv
- name: DIETClassifier
  epochs: 100
```
