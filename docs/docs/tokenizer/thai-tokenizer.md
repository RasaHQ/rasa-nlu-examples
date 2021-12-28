The `ThaiTokenizer` is a Rasa compatible tokenizer for Thai, using [`PyThaiNLP`](https://github.com/PyThaiNLP/pythainlp) under the hood.

In order to use the `ThaiTokenizer` the language **must** be set to `th` - no
other languages are supported by this tokenizer.

!!! note
    In order to use this tool you'll need to ensure the correct dependencies are installed.

    ```
    pip install "rasa_nlu_examples[thai] @ https://github.com/RasaHQ/rasa-nlu-examples.git"
    ```


## Configurable Variables

None

## Base Usage

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

If there are any issues with this tokenizer, please [let us know](https://github.com/RasaHQ/rasa-nlu-examples/issues).
