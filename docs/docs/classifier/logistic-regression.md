# LogisticRegressionClassifier

This intent classifier is based on the Logistic Regression Classifier from
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
This classifier only looks at sparse features extracted from the Rasa NLU
feature pipeline and is a *much* faster alternative to neural models like
[DIET](https://rasa.com/docs/rasa/components#dietclassifier-2).

## Configurable Variables

The classifier supports the same parameters as those that are listed in the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The only difference is:

- there is no `warm_start option`
- the default `class_weight` is "balanced"

## Base Usage

The configuration file below demonstrates how you might use the this component.
In this example we are extracting sparse features with two
CountVectorsFeaturizer instances, the first of which produces sparse
bag-of-words features, and the second which produces sparse
bags-of-character-ngram features.

Note that in the following example, setting the `class_weight` parameter to `None`
explicitly does have an effect because our default value for this paramter is "balanced".

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.classifiers.LogisticRegressionClassifier
  class_weight: None
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
- name: rasa_nlu_examples.classifiers.LogisticRegressionClassifier
- name: CRFEntityExtractor
```
