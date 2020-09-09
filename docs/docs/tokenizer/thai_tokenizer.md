The `ThaiTokenizer` is a Rasa compatible tokenizer for Thai, using [`PyThaiNLP`](https://github.com/PyThaiNLP/pythainlp) under the hood.

Once `pythainlp` is installed (e.g. via `pip install pythainlp`), it can be used in a Rasa configuration like below:

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