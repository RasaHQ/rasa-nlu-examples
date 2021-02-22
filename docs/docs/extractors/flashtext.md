#FlashTextEntityExtractor

This entity extractor uses the [flashtext](https://flashtext.readthedocs.io/en/latest/) library
to extract entities using [lookup tables](https://rasa.com/docs/rasa/nlu-training-data#lookup-tables).

This is similar to [RegexEntityExtractor](https://rasa.com/docs/rasa/components#regexentityextractor), but
different in a few ways:
1. `FlashTextEntityExtractor` takes only `lookups`, **not** regex patterns
2. `FlashTextEntityExtractor` matches using whitespace word boundaries. You cannot set `FlashTextEntityExtractor`
to match words regardless of boundaries.
3. `FlashTextEntityExtractor` is faster than `RegexEntityExtractor`

Also note that anything other than `[A-Za-z0-9_]` is considered a word boundary. To add more non-word boundaries
use the parameter `non_word_boundaries`

#Configurable Variables
- **case_sensitive**: whether to consider case when matching entities. `False` by default.
- **non_word_boundaries**: characters which shouldn't be considered word boundaries.

#Base Usage
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
  case_sensitive: True
  non_word_boundary:
  - "_"
  - ","
- name: DIETClassifier
  epochs: 100
```
You must include [lookup tables](https://rasa.com/docs/rasa/nlu-training-data#lookup-tables) in your NLU data. This
might look like:
```yaml
nlu:
- lookup: country
  examples: |
    - Afghanistan
    - Albania
    - ...
    - Zambia
    - Zimbabwe
```
In this example, anytime a user's utterance contains an exact match for a country from the lookup table above,
`FlashTextEntityExtractor` will extract this as an entity with type `country`. You should include a few examples with
this entity in your intent data, like so:

```yaml
- intent: inform_home_country
  examples: |
    - I am from [Afghanistan](country)
    - My family is from [Albania](country
```
