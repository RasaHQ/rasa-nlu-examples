# SparseNaiveBayesIntentClassifier

This intent classifier is based on the Bernoulli-variant of the Na&iuml;ve
Bayes classifier in
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html).
This classifier only looks at sparse features extracted from the Rasa NLU
feature pipeline and is a faster alternative to neural models like
[DIET](https://rasa.com/docs/rasa/components#dietclassifier-2). This model
requires that there be some sparse featurizers in your pipeleine. If you config
only has dense features it will throw an exception.

## Configurable Variables

- **alpha** (default: 1.0): Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
- **binarize** (default: 0.0): Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
- **fit_prior** (default: True): Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
- **class_prior** (default: None): Prior probabilities (as a list) of the classes. If specified the priors are not adjusted according to the data.

## Base Usage

The configuration file below demonstrates how you might use the this component.
In this example we are extracting sparse features with two
CountVectorsFeaturizer instances, the first of which produces sparse
bag-of-words features, and the second which produces sparse
bags-of-character-ngram features. We've also set the alpha smoothing parameter
to 0.1.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.classifiers.SparseNaiveBayesClassifier
  alpha: 0.1
```

Unlike [DIET](https://rasa.com/docs/rasa/components#dietclassifier-2), this
classifier only predicts intents. If you also need entity extraction, you will
have to add a separate entity extractor to your config. Below is an example
where we have included the CRFEntityExtractor to extract entities.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.classifiers.SparseNaiveBayesClassifier
  alpha: 0.1
- name: CRFEntityExtractor
```
