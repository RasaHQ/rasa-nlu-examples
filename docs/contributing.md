# Contributing Guide

We're open to contributions and there are many ways that you can make a contribution.

## Open an Issue

Did you find a bug? Please let us know! You can submit an issue [here](https://github.com/RasaHQ/rasa-nlu-examples/issues).

## Share an Observation

If the tools that we offer here turn out to be useful then we'd love to hear about it.
The research team will consider all feedback and we're especially keen to hear feedback
from non-English languages that try out some of the new embeddings. You can leave a message
either on [the github issue list](https://github.com/RasaHQ/rasa-nlu-examples/issues) or
on [the Rasa forum](forum.rasa.com/). Be sure to ping `koaning` on the forum if you mention
this project, he's the maintainer.

## Adding a new Component

There's a balance between allowing experimentation and maintaining all the code.
So we've come up with the following checklist.

1. If you want to contribute a new component please make an issue first so we can
discuss it. We want to prevent double work where possible and make sure it is
appropriate.
2. New tools that are added here need to be plausibly useful. For example,
we won't be able to accept a component that adds noise to the features.
3. Think about unit tests. We prefer to standardise unit tests as much
as possible but there may be specific things you'd like to check for.
