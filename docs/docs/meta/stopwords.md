This component can remove stopwords from the text. The idea is to filter
out words before the text is passed to the tokeniser. This way we prevent
many alignment mishaps between the tokens en the found entities.

## Configurable Variables

- **path**: path to a file that contains a stopword on each line

## Base Usage

Here's an example configuration.

```yaml
language: en

pipeline:
- name: rasa_nlu_examples.meta.StopWordRemover
  path: tests/data/stopwords/stopwords.txt
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 100
```

Note that you *must* place the `StopWordRemover` component before the tokeniser.
