# FasttextLanguageFallbackClassifier

This classifier uses [fasttext](https://fasttext.cc/docs/en/language-identification.html) to detect
if an unintended language is used. You can combine this tool together with [RulePolicy rules](https://rasa.com/docs/rasa/rules)
to handle out of scope responses more elegantly. Assuming that you're making an assistant to handle
English then you can send the user an appropriate response if this model predicts another language.

The tool should be able to detect 176 languages but the predictions won't be perfect. Especially
when the user sends short utterances we need to be careful. That is why this tool allows you to
specify a minimum number of characters and tokens before this model triggers an intent.

## Understanding the Tool

You're encouraged to play with the tool from a jupyter notebook so you can
understand what kinds of mistakes the model might make. To do that you'll
first want to download the `fasttext` library as well as the pretrained model.

```bash
python -m pip install fasttext
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
```

Then from a notebook you should be able to interact with the model by running:

```python
import fasttext
model = fasttext.load_model("lid.176.ftz")

txt = "i am speaking english"
proba_dict = {k: v for k, v in zip(*model.predict(txt, k=10))}
proba_dict
```

## Configurable Variables

- **expected_language** (required): the language that you expect to be predicted. If this language is predicted we won't trigger an intent.
- **intent_triggered** (required): the name of the intent to trigger when the model does not detect the expected language.
- **cache_dir** (required): specifies the folder where the pretrained model can be found.
- **model_file** (required): specifies the path to a pretrained model file, typically you'll want `lid.176.ftz`. See the [fasttext docs](https://fasttext.cc/docs/en/language-identification.html) for more info.
- **threshold** (default: 0.7): if the probability for your language is smaller than this threshold then we trigger the intent.
- **min_tokens** (default: 3): the minimum number of tokens that need to be in the utterance. If there's less tokens the language model is ignored because it is likely to be in-accurate.
- **min_chars** (default: 10): the minimum number of characters of text that need to be in the utterance. If there's less tokens the language model is ignored because it is likely to be in-accurate.
- **protected_intents** (default: `[]`): specifies a list of intent names that won't be overwritten

## Base Usage

The configuration file below demonstrates how you might use the this component. In this example
we've assumed that you've downloaded the lightweight `"lid.176.ftz"` model beforehand and that it
exists in the `downloaded` folder that is on the root path of your project.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 1
- name: rasa_nlu_examples.fallback.FasttextLanguageFallbackClassifier
  expected_language: en
  intent_triggered: non_english
  cache_dir: downloaded
  file: 'lid.176.ftz'
  min_chars: 5
  min_tokens: 2
  threshold: 0.3
  protected_intents: ["greet"]
```

To get the most out of this tool you also need to add a rule to a `rules.yml` file.
That way you can configure an appropriate action whenever a user is speaking in the
wrong language. That might look something like this:

```yaml
rules:

- rule: Ask the user to switch to speaking in English.
  steps:
  - intent: non_english
  - action: utter_non_english
```

For more information on rules, see the [docs](https://rasa.com/docs/rasa/rules).
