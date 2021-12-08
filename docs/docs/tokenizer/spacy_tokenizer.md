Rasa natively supports spaCy models that have a language model attached. But spaCy
also offers tokenizers without a model. We support these tokenisers with this
component.

!!! note
    In order to use this tool you'll need to ensure that spaCy is installed with Rasa.

    ```
    pip install rasa[spacy]
    ```

    You should also be aware that for certain languages extra dependencies are required.
    More information is given on the [spacy documentation](https://spacy.io/usage/models/#languages).


## Configurable Variables

- **lang**: the two-letter [abbreviation](https://spacy.io/usage/models/#languages) of the language you want to use.

## Base Usage

Once downloaded it can be used in a Rasa configuration, like below;

```yaml
language: en

pipeline:
- name: rasa_nlu_examples.tokenizers.BlankSpacyTokenizer
  lang: "en"
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 100
```
