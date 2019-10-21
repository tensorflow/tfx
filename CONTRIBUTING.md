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

By default, only unit tests are executed. Tests marked as `end_to_end` will be
skipped.

## Testing local change
To test local change, you will need to install code into virtualenv in editable
mode:

```
pushd <your_source_dir>
pip install -e .
```


# Check Pending Changes
Each change being worked on internally will have a pending PR, which will be
automatically closed once internal change is submitted. You are welcome to
checkout the PR branch to observe the behavior.

# Merging External Contributions
External contributions can be submitted normally as a Github PR.

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult GitHub Help for more
information on using pull requests.

After a pull request is approved, we merge it. Note our merging process differs
from GitHub in that we pull and submit the change into an internal version
control system. This system automatically pushes a git commit to the GitHub
repository (with credit to the original author) and closes the pull request.

A TFX engineer will monitor all public PRs and look at the issue within 5
business days to perform initial triage.

## Python 2 and type hints
Starting from TFX 0.14, all python source code on github includes
[Python 3 type hints](https://docs.python.org/3.5/library/typing.html). Python
3.6 (or above) is the recommended version to develop this project.

The Python 2 version of [tfx](https://pypi.org/project/tfx/) PyPI package has
all type hints stripped and is the easiest way for python 2 users.

Python 2 support will be deprecated in 2020, due to
[python 2 not supported beyong 2020](https://www.python.org/dev/peps/pep-0373/).

If you must use Python 2 with source code:

* run `python tfx/scripts/strip_type_hints.py` to strip all hints in the repo:
this requires you to install the
[strip-hints](https://pypi.org/project/strip-hints/) PyPI package.
