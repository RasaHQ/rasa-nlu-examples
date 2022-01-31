# FlashTextEntityExtractor

!!! note
    If you want to use this component, be sure to either install flashtext manually
    or use our convenience installer.

    ```
    python -m pip install "rasa_nlu_examples[flashtext] @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
    ```

This entity extractor uses the [flashtext](https://flashtext.readthedocs.io/en/latest/) library
to extract entities.

This is similar to [RegexEntityExtractor](https://rasa.com/docs/rasa/components#regexentityextractor), but
different in a few ways:

1. `FlashTextEntityExtractor` uses token-matching to find entities, **not** regex patterns
2. `FlashTextEntityExtractor` matches using whitespace word boundaries. You cannot set it
to match words regardless of boundaries.
3. `FlashTextEntityExtractor` is *much* faster than `RegexEntityExtractor`. This is especially true
for large lookup tables.

Also note that anything other than `[A-Za-z0-9_]` is considered a word boundary. To add more non-word boundaries
use the parameter `non_word_boundaries`

## Configurable Variables

- **path**: the path to the lookup text file
- **entity_name**: the name of the entity to attach to the message
- **case_sensitive**: whether to consider case when matching entities. `False` by default.
- **non_word_boundaries**: characters which shouldn't be considered word boundaries.
- **encoding**: the name of the encoding used to read the lookup text file. 

## Base Usage

The configuration below is an example of how you might use`FlashTextEntityExtractor`.
```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.extractors.FlashTextEntityExtractor
  case_sensitive: False
  path: path/to/file.txt
  entity_name: country
- name: DIETClassifier
  epochs: 100
```
You must include a plain text file that contains the tokens to detect.
Such a file might look like:

```yaml
Afghanistan
Albania
...
Zambia
Zimbabwe
```

In this example, anytime a user's utterance contains an exact match for a country,
`FlashTextEntityExtractor` will extract this as an entity with type `country`. You should include a few examples with
this entity in your intent data, like so:

```yaml
- intent: inform_home_country
  examples: |
    - I am from [Afghanistan](country)
    - My family is from [Albania](country)
```
