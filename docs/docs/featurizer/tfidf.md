This featurizer is a *sparse* featurizer. It builds on the
[scikit-learn implementation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn-feature-extraction-text-tfidfvectorizer)
to convert text into sparse features that take the frequency of words into account. If we
were to feed the direct count data directly to a classifier very frequent terms might shadow
the frequencies of rarer, but potentially more interesting words.


## Configurable Variables

- **analyzer**: determines how tokens are split. possible choices are `word`, `char` and `char_wb`, default is `word`.
- **min_ngram**: the lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
- **max_ngram**: the upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.

## Base Usage

The configuration file below demonstrates how you might use the TfIdfFeaturizer featurizer.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.sparse.TfIdfFeaturizer
  min_ngram: 1
  max_ngram: 2
- name: DIETClassifier
  epochs: 100
```
