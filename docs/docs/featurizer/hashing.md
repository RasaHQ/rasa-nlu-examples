This page discusses some properties of the `HashingFeaturizer`.
Note that this featurizer is a *sparse* featurizer.

The featurizer is a wrapper a around scikit-learn's
[HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html).
It uses the "hashing trick" to transform input texts to a sparse vector by mapping each token to a column index using a
fixed hash function. The featurizer has no state and cannot be trained. For a small number of columns (defined by the
`n_features` parameter), hash coalitions are more likely, meaning that two words can get mapped to the
same index.

## Configurable Variables

- **n_features**: the number of columns the input is mapped to.
- **analyzer**: determines how tokens are split. possible choices are `word`, `char` and `char_wb`.
- **lowercase**: convert input strings to lowercase.
- **strip_accents**: remove accents using one of the methods `ascii` or `unicode`.
- **stop_words**: filter by a list of stop words.
- **min_ngram**: the lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
- **max_ngram**: the upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
- **norm**: the normalization applied to each row vector (options are `l1`, `l2` or `null`).
- **binary**: if `True`, all non-zero elements are mapped to `1`, instead of absolute counts.
- **alternate_sign**: apply the sign of the hashing function in order to reduce the effect of hash coalitions.

## Base Usage

The configuration file below demonstrates how you might use the hashing featurizer.

```yaml
pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: rasa_nlu_examples.featurizers.sparse.HashingFeaturizer
  n_features: 1024
- name: DIETClassifier
  epochs: 100
```

## Combining several hashing featurizers

In order to use a combination of several hash functions,  multiple `HashingFeaturizer` instances can be
added to the pipeline. However, note that since the hash function is deterministic, one needs to set a
varying number of `n_features` for each. Otherwise one would end up with the same sparse vector being
concatenated multiple times. See the discussion
[here](https://github.com/RasaHQ/rasa-nlu-examples/issues/148#issuecomment-975335531)
how this, in combination with `DIETClassifier`, is related to
[Bloom embeddings](https://support.prodi.gy/t/can-you-explain-how-exactly-hashembed-works/564/2).

```yaml
pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: rasa_nlu_examples.featurizers.sparse.HashingFeaturizer
  n_features: 1021
- name: rasa_nlu_examples.featurizers.sparse.HashingFeaturizer
  n_features: 1022
- name: rasa_nlu_examples.featurizers.sparse.HashingFeaturizer
  n_features: 1023
- name: rasa_nlu_examples.featurizers.sparse.HashingFeaturizer
  n_features: 1024
- name: DIETClassifier
  epochs: 100
```
