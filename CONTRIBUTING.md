# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose. Consult GitHub Help for more
information on using pull requests.

## Community Guidelines
This project follows[Google's Open Source Community Guidelines](
https://opensource.google.com/conduct/).

# Contributing Guidelines

## Running Unit Tests
We recommend using virtualenv and Python 3.5 and above for development. You can
invoke all python unit tests:

```
python setup.py test
```

# Check Pending Changes
Each change being worked on internally will have a pending PR, which will be
automatically closed once internal change is submitted. You are welcome to
checkout the PR branch to observe the behavior.

## Python 2 and type hints
Starting from TFX 0.14, all python source code on github includes
[Python 3 type hints](https://docs.python.org/3.5/library/typing.html). Python
3.5 (or above) is the recommended version to develop this project.

The Python 2 version of [tfx](https://pypi.org/project/tfx/) PyPI package has
all type hints stripped and is the easiest way for python 2 users.

If you must use Python 2 with source code:

* run `python tfx/scripts/strip_type_hints.py` to strip all hints in the repo:
this requires you to install the
[strip-hints](https://pypi.org/project/strip-hints/) PyPI package.
