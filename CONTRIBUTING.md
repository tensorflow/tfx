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

Proper code review should include proper unit tests with them. Please see
[Testing Conventions](##Testing Conventions) for our conventions on how to write
proper unit tests.

TODO(zhitaoli): Configure a CI/CD on Github so that each PR submission is tested
before merged.

## Community Guidelines
This project follows[Google's Open Source Community Guidelines](
https://opensource.google.com/conduct/).

# Contributing Guidelines

At this point, TFX only supports Python 3 on Linux and MacOS. Please use one of
these operation system for development and testing.

If Python 3.5 is used, our usage of type hints requires at least 3.5.3.

## Testing Conventions

All python unit tests in this repo is based on Tensorflow's
[tf.test.TestCase](https://www.tensorflow.org/api_docs/python/tf/test/TestCase),
which is a subclass of
[py-absl TestCase](https://github.com/abseil/abseil-py/blob/06edd9c20592cec39178b94240b5e86f32e19768/absl/testing/absltest.py#L523).

We have several types of tests in this repo: * Unit tests for source code; * End
to end tests (filename ends with `_e2e_test.py`): some of this also runs with
external environments.

## Testing local change
To test local change, you will need to install code into virtualenv in editable
mode:

```shell
pushd <your_source_dir>
pip install -e .[all]   # the [all] suffix includes additional packages for test
```

## Running Unit Tests

At this point all unit tests are safe to run externaly. We are working on
porting the end to end tests.

Each test can just be invoked with `python`. To invoke all unit tests:

```shell
find . -name '*_test.py' | grep -v e2e | xargs -I {} python {}
```

By default, only unit tests are executed. Tests marked as `end_to_end` will be
skipped.

# Check Pending Changes
Each change being worked on internally will have a pending PR, which will be
automatically closed once internal change is submitted. You are welcome to
checkout the PR branch to observe the behavior.

# Merging External Contributions

External contributions can be submitted normally as a GitHub PR.

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult GitHub Help for more
information on using pull requests.

After a pull request is approved, we merge it. Note our merging process differs
from GitHub in that we pull and submit the change into an internal version
control system. This system automatically pushes a git commit to the GitHub
repository (with credit to the original author) and closes the pull request.

It is recommended to reach out to TFX engineers to establish consensus on the
tentative design before submitting PRs to us, as well as finding a potential
reviewer.

For public PRs which do not have a preassigned reviewer, a TFX engineer will
monitor them and perform initial triage within 5 business days. But such
contributions should be trivial (i.e, documentation fixes).
