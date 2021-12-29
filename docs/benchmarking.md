# Benchmarking Guide

This is a small guide that will explain how you can use the tools in this library to run benchmarks.

As an example project we'll use [the Sara demo](https://github.com/rasahq/rasa-demo).

First you'll need to install the project. An easy way to do this is via pip;

```bash
pip install git+https://github.com/RasaHQ/rasa-nlu-examples
```

You should now be able to run configuration files with NLU components
from this library. You can glance over some examples below.


=== "Basic Config"
    Here's a very basic configuartion file.
    <pre lang="yaml"><code>language: en
    pipeline:
    - name: WhitespaceTokenizer
    - name: CountVectorsFeaturizer
      OOV_token: oov.txt
      analyzer: word
    - name: CountVectorsFeaturizer
      analyzer: char_wb
      min_ngram: 1
      max_ngram: 4
    - name: DIETClassifier
      epochs: 200
    </code></pre>

    Assuming this file is named `basic-config.yml` you can run this pipeline as a benchmark by running this
    command from the project directory;

    <pre><code>rasa test nlu --config basic-config.yml \
              --cross-validation --runs 1 --folds 2 \
              --out gridresults/basic-config
    </code></pre>

    This will generate output in the `gridresults/basic-config` folder.

=== "Basic Byte-Pair"
    Here's the same basic configuration but now with dense features added.
    <pre><code>language: en
    pipeline:
    - name: WhitespaceTokenizer
    - name: CountVectorsFeaturizer
      OOV_token: oov.txt
      analyzer: word
    - name: CountVectorsFeaturizer
      analyzer: char_wb
      min_ngram: 1
      max_ngram: 4
    - name: rasa_nlu_examples.featurizers.dense.BytePairFeaturizer
      lang: en
      vs: 1000
      dim: 25
    - name: DIETClassifier
      epochs: 200
    </code></pre>

    Assuming this file is named `basic-bytepair-config.yml` you can run it as a benchmark by running this
    command from the project directory;

    <pre><code>rasa test nlu --config basic-bytepair-config.yml \
              --cross-validation --runs 1 --folds 2 \
              --out gridresults/basic-bytepair-config
    </code></pre>

    This will generate output in the `gridresults/basic-bytepair-config` folder.

=== "Medium Byte-Pair"
    We've now increased the vocabulary size and dimensionality.
    <pre><code>language: en
    pipeline:
    - name: WhitespaceTokenizer
    - name: CountVectorsFeaturizer
      OOV_token: oov.txt
      analyzer: word
    - name: CountVectorsFeaturizer
      analyzer: char_wb
      min_ngram: 1
      max_ngram: 4
    - name: rasa_nlu_examples.featurizers.dense.BytePairFeaturizer
      lang: en
      vs: 10000
      dim: 100
    - name: DIETClassifier
      epochs: 200
    </code></pre>

    Assuming this file is named `medium-bytepair-config.yml` you can run it as a benchmark by running this
    command from the project directory;

    <pre><code>rasa test nlu --config medium-bytepair-config.yml \
              --cross-validation --runs 1 --folds 2 \
              --out gridresults/medium-bytepair-config
    </code></pre>

    This will generate output in the `gridresults/medium-bytepair-config` folder.

=== "Large Byte-Pair"
    We've now grabbed the largest English Byte-Pair embeddings available.
    <pre><code>language: en
    pipeline:
    - name: WhitespaceTokenizer
    - name: CountVectorsFeaturizer
      OOV_token: oov.txt
      analyzer: word
    - name: CountVectorsFeaturizer
      analyzer: char_wb
      min_ngram: 1
      max_ngram: 4
    - name: rasa_nlu_examples.featurizers.dense.BytePairFeaturizer
      lang: en
      vs: 200000
      dim: 300
    - name: DIETClassifier
      epochs: 200
    </code></pre>

    Assuming this file is named `large-bytepair-config.yml` you can run this benchmark by running this
    command from the project directory;

    <pre><code>rasa test nlu --config large-bytepair-config.yml \
              --cross-validation --runs 1 --folds 2 \
              --out gridresults/large-bytepair-config
    </code></pre>

    This will generate output in the `gridresults/large-bytepair-config` folder.


## Final Reminder

We should remember that these tools are experimental in nature. We want this repository to be a place
where folks can share their nlu components and experiment, but this also means that we don't want to
suggest that these tools are state of the art. You always need to check if these tools work for your
pipeline. The components that we host here may very well lag behind Rasa Open Source too.
