# DateparserEntityExtractor

!!! note
    If you want to use this component, be sure to either install flashtext manually
    or use our convenience installer.

    ```
    python -m pip install "rasa_nlu_examples[dateparser] @ git+https://github.com/RasaHQ/rasa-nlu-examples.git"
    ```

## What does it do?

This entity extractor uses the [dateparser](https://dateparser.readthedocs.io/en/latest/) to extract
entities that resemble dates. You can get a demo by running the code below.

```python
from rasa.shared.nlu.training_data.message import Message
from rasa_nlu_examples.extractors.dateparser_extractor import DateparserEntityExtractor
from rich import print

msg = Message.build("hello tomorrow, goodbye yesterday",)
extractor = DateparserEntityExtractor({})
extractor.process(msg)
print(msg.as_dict_nlu())
```

This will parse the following information.

```python
{
    'text': 'hello tomorrow, goodbye yesterday',
    'entities': [
        {
            'entity': 'DATETIME_REFERENCE',
            'start': 6,
            'end': 14,
            'value': 'tomorrow',
            'parsed_date': '2021-06-05 11:50:10.502082',
            'confidence': 1.0,
            'extractor': 'DateparserEntityExtractor'
        },
        {
            'entity': 'DATETIME_REFERENCE',
            'start': 24,
            'end': 33,
            'value': 'yesterday',
            'parsed_date': '2021-06-03 11:50:10.503160',
            'confidence': 1.0,
            'extractor': 'DateparserEntityExtractor'
        }
    ]
}
```

Note that we add an extra `parsed_date` key to the entity dictionary here. Another
benefit of `dateparser` is that it also contains rules for Non-English languages. Here
is a Dutch example.

```python
{
    'text': 'ik wil een pizza bestellen voor morgen',
    'entities': [
        {
            'entity': 'DATETIME_REFERENCE',
            'start': 32,
            'end': 38,
            'value': 'morgen',
            'parsed_date': '2021-06-05 11:50:10.708588',
            'confidence': 1.0,
            'extractor': 'DateparserEntityExtractor'
        }
    ]
}
```

It's also possible to configure the `DateparserEntityExtractor` to prefer dates in the
future or in the past. That way, if somebody talks about `Thursday` can be picked up as
*next* Thursday, allowing us to still parse out a date.

### "Future" Results

This ran on Friday the 4th of June, 2021.

```python
{
    'text': 'i want a pizza thursday',
    'entities': [
        {
            'entity': 'DATETIME_REFERENCE',
            'start': 15,
            'end': 23,
            'value': 'thursday',
            'parsed_date': '2021-06-10 00:00:00',
            'confidence': 1.0,
            'extractor': 'DateparserEntityExtractor'
        }
    ]
}
```

### "Past" Results

This ran on Friday the 4th of June, 2021.

```python
{
    'text': 'i want to buy a pizza thursday',
    'entities': [
        {
            'entity': 'DATETIME_REFERENCE',
            'start': 22,
            'end': 30,
            'value': 'thursday',
            'parsed_date': '2021-06-03 00:00:00',
            'confidence': 1.0,
            'extractor': 'DateparserEntityExtractor'
        }
    ]
}
```

## Configurable Variables

- **languages**: pass a list of languages that you want the parser to focus on, can be `None` but this setting is likely to overfit on English assumptions
- **prefer_dates_from**: can be either "future", "past" or `None`
- **relative_base**: can be a datestring that represents a reference date, this is useful when a user mentions "tomorrow", default `None` points to todays date

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
- name: DIETClassifier
  epochs: 100
- name: rasa_nlu_examples.extractors.DateparserEntityExtractor
  languages: ["en", "nl", "es"]
  prefer_dates_from: "future"
```

Note that this entity extractor completely ignores the tokeniser. There might also be
overlap with enities from other engines, like DIET and spaCy.

## Relative Base Usage

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 100
- name: rasa_nlu_examples.extractors.DateparserEntityExtractor
  languages: ["en", "nl", "es"]
  prefer_dates_from: "future"
  relative_base: "2020-01-01"
```
