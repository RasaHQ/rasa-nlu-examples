The `SemanticMapFeaturizer` is an experimental *sparse* featurizer developed by Rasa.
It can only be used in combination with pre-trained embedding files, which you can
find [here](). Please refer to our [blog post]() for more details.

## Configurable Variables

- **pretrained_semantic_map**: Path to downloaded/saved semantic map embeddings (the unpacked json file)
- **pooling**: The pooling operation to use for the sentence features (`sum`, `mean`, or `merge`)

## Basic Usage

The configuration file below demonstrates how you might use the semantic map embeddings. In this example
we're building a pipeline for the English language and we're assuming that you've saved embeddings upfront
as `saved/beforehand/rasa-sme-wikipedia-en-64x64-v20201120.json`.

```yaml
language: en

pipeline:
  - name: WhitespaceTokenizer
  - name: LexicalSyntacticFeaturizer
  - name: rasa_nlu_examples.featurizers.sparse.SemanticMapFeaturizer
    pretrained_semantic_map: "saved/beforehand/rasa-sme-wikipedia64x64-en-v20201120.json"
  - name: DIETClassifier
    epochs: 100
```
