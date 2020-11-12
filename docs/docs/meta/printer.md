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

When you now interact with your model via `rasa shell` you will see pretty information appear about the state
of the `Message` object. It might look something like this;


<style>
.r1 {font-weight: bold}
.r2 {color: #008000}
.r3 {color: #000080; font-weight: bold}
</style>
<code>
        <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span class="r1">{</span>
    <span class="r2">'text'</span>: <span class="r2">'rasa nlu examples'</span>,
    <span class="r2">'intent'</span>: <span class="r1">{</span><span class="r2">'name'</span>: <span class="r2">'out_of_scope'</span>, <span class="r2">'confidence'</span>: <span class="r3">0.4313829839229584</span><span class="r1">}</span>,
    <span class="r2">'entities'</span>: <span class="r1">[</span>
        <span class="r1">{</span>
            <span class="r2">'entity'</span>: <span class="r2">'proglang'</span>,
            <span class="r2">'start'</span>: <span class="r3">0</span>,
            <span class="r2">'end'</span>: <span class="r3">4</span>,
            <span class="r2">'confidence_entity'</span>: <span class="r3">0.42326217889785767</span>,
            <span class="r2">'value'</span>: <span class="r2">'rasa'</span>,
            <span class="r2">'extractor'</span>: <span class="r2">'DIETClassifier'</span>
        <span class="r1">}</span>
    <span class="r1">]</span>,
    <span class="r2">'text_tokens'</span>: <span class="r1">[</span><span class="r2">'rasa'</span>, <span class="r2">'nlu'</span>, <span class="r2">'examples'</span><span class="r1">]</span>,
    <span class="r2">'intent_ranking'</span>: <span class="r1">[</span>
        <span class="r1">{</span><span class="r2">'name'</span>: <span class="r2">'out_of_scope'</span>, <span class="r2">'confidence'</span>: <span class="r3">0.4313829839229584</span><span class="r1">}</span>,
        <span class="r1">{</span><span class="r2">'name'</span>: <span class="r2">'goodbye'</span>, <span class="r2">'confidence'</span>: <span class="r3">0.2445288747549057</span><span class="r1">}</span>,
        <span class="r1">{</span><span class="r2">'name'</span>: <span class="r2">'bot_challenge'</span>, <span class="r2">'confidence'</span>: <span class="r3">0.23958507180213928</span><span class="r1">}</span>,
        <span class="r1">{</span><span class="r2">'name'</span>: <span class="r2">'greet'</span>, <span class="r2">'confidence'</span>: <span class="r3">0.04896979033946991</span><span class="r1">}</span>,
        <span class="r1">{</span><span class="r2">'name'</span>: <span class="r2">'talk_code'</span>, <span class="r2">'confidence'</span>: <span class="r3">0.035533301532268524</span><span class="r1">}</span>
    <span class="r1">]</span>,
    <span class="r2">'dense'</span>: <span class="r1">{</span>
        <span class="r2">'sequence'</span>: <span class="r1">{</span><span class="r2">'shape'</span>: <span class="r1">(</span><span class="r3">3</span>, <span class="r3">25</span><span class="r1">)</span>, <span class="r2">'dtype'</span>: dtype<span class="r1">(</span><span class="r2">'float32'</span><span class="r1">)}</span>,
        <span class="r2">'sentence'</span>: <span class="r1">{</span><span class="r2">'shape'</span>: <span class="r1">(</span><span class="r3">1</span>, <span class="r3">25</span><span class="r1">)</span>, <span class="r2">'dtype'</span>: dtype<span class="r1">(</span><span class="r2">'float32'</span><span class="r1">)}</span>
    <span class="r1">}</span>,
    <span class="r2">'sparse'</span>: <span class="r1">{</span>
        <span class="r2">'sequence'</span>: <span class="r1">{</span><span class="r2">'shape'</span>: <span class="r1">(</span><span class="r3">3</span>, <span class="r3">1780</span><span class="r1">)</span>, <span class="r2">'dtype'</span>: dtype<span class="r1">(</span><span class="r2">'float64'</span><span class="r1">)</span>, <span class="r2">'stored_elements'</span>: <span class="r3">67</span><span class="r1">}</span>,
        <span class="r2">'sentence'</span>: <span class="r1">{</span><span class="r2">'shape'</span>: <span class="r1">(</span><span class="r3">1</span>, <span class="r3">1756</span><span class="r1">)</span>, <span class="r2">'dtype'</span>: dtype<span class="r1">(</span><span class="r2">'int64'</span><span class="r1">)</span>, <span class="r2">'stored_elements'</span>: <span class="r3">32</span><span class="r1">}</span>
    <span class="r1">}</span>
<span class="r1">}</span>
</pre>
    </code>
