Here's an example configuration file that demonstrates how the custom printer component works.
You can find a tutorial on the component [here](https://blog.rasa.com/custom-printer-component/).

## Configurable Variables

- **alias**: gives an extra name to the component and adds an extra message that is printed

## Base Usage

When running this example you'll notice that what the effect is of the `CountVectorsFeaturizer`.
You should see print statements appear when you talk to the assistant.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: rasa_nlu_examples.meta.Printer
  alias: before count vectors
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.meta.Printer
  alias: after count vectors
- name: DIETClassifier
  epochs: 100
```
