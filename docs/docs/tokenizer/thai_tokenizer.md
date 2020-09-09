The `ThaiTokenizer` is a Rasa compatible tokenizer for Thai, using [`PyThaiNLP`](https://github.com/PyThaiNLP/pythainlp) under the hood.

In order to use the `ThaiTokenizer` the language **must** be set to `th` - no other languages are supported by this tokenizer.

The `ThaiTokenizer` can be used in a Rasa configuration like below:

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
    epochs: 100
```