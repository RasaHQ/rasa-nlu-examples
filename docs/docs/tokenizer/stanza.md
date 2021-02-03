The [Stanza project](https://stanfordnlp.github.io/stanza/) from Stanford supports tokenizers, lemmatizers as
well as part of speech detection for many languages that are not supported by spaCy. You can find the available
languages [here](https://stanfordnlp.github.io/stanza/available_models.html).

!!! note
    In order to use this tool you'll need to ensure the correct dependencies are installed.

    ```
    pip install "rasa_nlu_examples[stanza] @ https://github.com/RasaHQ/rasa-nlu-examples.git"
    ```

## Model Download

To use a Stanza model you'll first need to download it. This can be done from python.

```python
import stanza
# download English model in the ~/stanza_resources dir
stanza.download('en', dir='~/stanza_resources')
```

## Configurable Variables

- **lang**: then two-letter abbreprivation of the language you want to use
- **cache_dir**: pass it the name of the directory where you've downloaded/saved the embeddings

## Base Usage

Once downloaded it can be used in a Rasa configuration, like below;

```yaml
language: en

pipeline:
- name: rasa_nlu_examples.tokenizers.StanzaTokenizer
  lang: "en"
  cache_dir: "~/stanza_resources"
- name: LexicalSyntacticFeaturizer
  "features": [
    ["low", "title", "upper"],
    ["BOS", "EOS", "low", "upper", "title", "digit", "pos"],
    ["low", "title", "upper"],
  ]
- name: CountVectorsFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 100
```

One thing to note here is that the `LexicalSyntacticFeaturizer` will be able to pick up
the "pos" information with the `StanzaTokenizer` just like you're able to do that with spaCy.
The `CountVectorsFeaturizer` is able to pick up the `lemma` features that are generated.
