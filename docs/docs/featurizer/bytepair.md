This featurizer is a *dense* featurizer. If you're interested in learning how these
work you might appreciate reading[the original article](http://www.lrec-conf.org/proceedings/lrec2018/pdf/1049.pdf).
Recognition should be given to Benjamin Heinzerling and Michael Strube for making these available.

A main feature of these types of embeddings is that they are relatively lightweight but also
that they're availability in many languages. BytePair embeddings exist for **277** languages that are
pretrained on wikipedia.

More information on these embeddings can be found [here](https://bpemb.h-its.org/). When you scroll down you will
notice a large of languages that are available. Here's some examples from that list that give a detailed view of available
vectors:

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
- **vs_fallback**: if set to True and the given vocabulary size can't be loaded for the given model, the closest size is chosen, default=`True`

## Base Usage

The configuration file below demonstrates how you might use the BytePair embeddings. The component
will automatically download the required embeddings and save them in `~/.cache`. Both the embeddings
as well as a model file will be saved.

```yaml
language: en

pipeline:
- name: WhitespaceTokenizer
- name: LexicalSyntacticFeaturizer
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: rasa_nlu_examples.featurizers.dense.BytePairFeaturizer
  lang: en
  vs: 1000
  dim: 25
- name: DIETClassifier
  epochs: 100
```
