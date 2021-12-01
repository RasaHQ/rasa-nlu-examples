This featurizer is a *sparse* featurizer.

A main feature of these features is that you can use a pretrained BytePair tokeniser
to encode the received text. Since BytePair embeddings exist for **277** languages
all of which also have a pretrained tokeniser we figured it worth sharing.

The tokeniser won't actually cause tokens to appear in Rasa. Rather, it applies
a tokenisation trick before doing a countvector-encoding.

```
"talking about geology" -> '▁talk ing ▁about ▁ge ology'
```

Instead of encoding three words, this featurizer would encode 5 subtokens. Note
that we also create features for each token. So the token "talking" will get the
features for "_talk" and "ing".

More information on the available models can be found [here](https://bpemb.h-its.org/).
When you scroll down you will  notice a large of languages that are available. Here's
some examples from that list that give a detailed view of available languages:

- [Abkhazian](https://bpemb.h-its.org/ab/)
- [Zulu](https://bpemb.h-its.org/zu/)
- [English](https://bpemb.h-its.org/en/)
- [Hindi](https://bpemb.h-its.org/hi/)
- [Chinese](https://bpemb.h-its.org/zh/)
- [Esperanto](https://bpemb.h-its.org/eo/)
- [Multi Language](https://bpemb.h-its.org/multi/)

## Configurable Variables

- **lang**: specifies the lanuage that you'll use, default = `"en"`
- **dim**: specifies the dimension of the subword embeddings, default = `25`,
- **vs**: specifies the vocabulary size of the segmentation model, default = `1000`,
- **cache_dir**: specifies the folder in which downloaded BPEmb files will be cached, default = `~/.cache/bpemb`
- **model_file**: specifies the path to a custom model file, default=`None`,

## Base Usage

The configuration file below demonstrates how you might use the BytePair embeddings. In this example
we're not using any cached folders and the library will automatically download the correct embeddings
for you and save them in `~/.cache`. Both the embeddings as well as a model file will be saved.

Also note that we recommend keeping the vocabulary size small if you're interested
in spelling robustness as well as token featurization. Large vocabularies correspond
with words being encoded as opposed to subwords.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.SparseBytePairFeaturizer
  lang: en
  vs: 1000
- name: DIETClassifier
  epochs: 100
```

## Cached Usage

If you're using pre-downloaded embedding files (in docker you might have this on a mounted disk)
then you can prevent a download from happening. We'll be doing that in the example below.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.SparseBytePairFeaturizer
  lang: en
  vs: 1000
  cache_dir: "tests/data"
- name: DIETClassifier
  epochs: 100
```

Note that in this case we expect a file to be present in the `tests/data/en` directory;

- `en.wiki.bpe.vs10000.model`

You can also overwrite the names of these files via the `model_file` setting. But it
is preferable to stick to the library naming convention. Also note that if you use
the `model_file` setting that you must provide full filepaths and that the
`cache_dir` will be ignored.
