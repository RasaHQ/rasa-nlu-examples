# Contributing Guide

## Ways you can Contribute

We're open to contributions and there are many ways that you can make one.

- You can suggest new features.
- You can help review new features.
- You can submit new components.
- You can let us know if there are bugs.
- You can let us know if the components in this library help you.

### Open an Issue

You can submit an issue [here](https://github.com/RasaHQ/rasa-nlu-examples/issues). Issues
allow us to keep track of a conversation about this repository and it is the preferred
communication channel for bugs related to this project.

### Suggest a New Feature

This project started because we wanted to offer support for word embeddings
for more languages. The first feature we added was support for FastText,
which offers embeddings for 157 languages. We later received a contribution from a community member for BytePair embeddings, which offers support for 275 languages. We weren't aware of these embeddings but we were exited to support more languages.

Odds are that there are many more tools out there that the maintainers of this project aren't
aware of yet. There may very well be more embeddings, tokenziers, lemmatizers and models
that we're not aware of but can help Rasa developers make better assistants.

The goal of this project is to support more of these sorts of tools for Rasa users. You
can help out the project just by letting us know if there's an integration missing. If you
do not have the time to contribute a component yourself then you can still contribute to the
effort by letting us know what components might help you make a better assistant.

### Share an Observation

If the tools that we offer here turn out to be useful then we'd love to hear about it.
We're also interested in hearing if these tools don't work for your usecase.
Any feedback will be shared with the research team at Rasa. We're especially keen to hear feedback on the performance of the word embeddings that we host here. You can leave a message
either on [the github issue list](https://github.com/RasaHQ/rasa-nlu-examples/issues) or
on [the Rasa forum](forum.rasa.com/). Be sure to ping **koaning** on the forum if you mention
this project, he's the main maintainer.

### Adding a new Component

There's a balance between allowing experimentation and maintaining all the code.
This is why we've come up with a checklist that you should keep in mind before
you're submitting code.

1. If you want to contribute a new component please make an issue first so we can
discuss it. We want to prevent double work where possible and make sure the proposed
component is appropriate for this repository.
2. New tools that are added here need to be plausibly useful in a real life scenario.
For example, we won't be able to accept a component that adds noise to the features.
3. Think about unit tests. We prefer to standardise unit tests as much
as possible but there may be specific things you'd like to check for.
4. Ask for help! Feel free to ping the **koaning** on this repository if you're having
trouble with an implementation or appreciate guidance on a topic. He'll gladly
help you.

## Testing

We run automated tests via GitHub actions but you can also run all the checking mechanisms locally.
To run the tests locally you'll need to run a few scripts beforehand.

```
python tests/scripts/prepare_fasttext.py
python tests/scripts/prepare_stanza.py
```

This will prepare the filesystem for testing. We do this to prevent the need of downloading
very large word embeddings locally and in CI. Fasttext can be 6-7 GB and we don't want to pull such a
payload at every CI step. You can also prepare files locally by installing all dependencies
via the `Makefile`.

```
make install
```

You can also run all style and type checking mechanisms locally via the `Makefile`.

```
make check
```
