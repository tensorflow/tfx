# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Requests For Comment

TFX is an open-source project and we strongly encourage active participation
by the ML community in helping to shape TFX to meet or exceed their needs. An
important component of that effort is the RFC process.  Please see the listing
of [current and past TFX RFCs](RFCs.md). Please see the
[TensorFlow Request for Comments (TF-RFC)](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md)
process page for information on how community members can contribute.

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
[Testing Conventions](#testing-conventions) for our conventions on how to write
proper unit tests.

TODO(zhitaoli): Configure a CI/CD on Github so that each PR submission is tested
before merged.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

# Contributing Guidelines

At this point, TFX only supports Python 3 (up to version 3.7) on Linux and
MacOS. Please use one of these operation system for development and testing.

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

To test local change, first you have to install
[Bazel](https://docs.bazel.build/versions/master/install.html), which powers the
protobuf stub code generation. Check whether Bazel is installed and executable:

```shell
bazel --version
```

After installing Bazel, you can move to the cloned source directory.

```shell
pushd <your_source_dir>
```

TFX has many dependent family libraries like TensorFlow Data Validation and
TensorFlow Model Analysis. Sometimes, TFX uses their most recent API changes
before published. So it is safer to use nightly versions of those libraries when
you develop TFX. You have to set property depdendency using
`TFX_DEPENDENCY_SELECTOR` environment variable, and supply our nightly package
index URL when installing TFX.

> NOTE: Please use the latest version of `pip` (Possibly after 21.0) before
> running following commands. TFX works well with the new dependency resolver
> from TFX 0.30.0. `pip install --upgrade pip` should upgrade pip.

You can install TFX source code in a virtual environment in editable (`-e`)
mode, which will pick up your local changes immediately without re-installing
every time.

```shell
export TFX_DEPENDENCY_SELECTOR=NIGHTLY

# You might need to install additional packages to run all end-to-end tests.
# To run all tests, use [test] extra requirements which includes all
# dependencies including airflow and kfp. If you want to test a specific
# orchestrator only, use [airflow] or [kfp]. (Beam and Local orchestators can
# be run without any extra dependency.) For example,
# $ pip install -e .[kfp] -i https://pypi-nightly.tensorflow.org/simple
pip install -e . -i https://pypi-nightly.tensorflow.org/simple
```

Alternatively, you can also build all TFX family libraries from github source
although it takes quite long.

```shell
export TFX_DEPENDENCY_SELECTOR=GIT_MASTER
pip install -e .
```

You can read more description on
[our dependency definition](https://github.com/tensorflow/tfx/blob/981d28e6d83a44d48cf070c28807fdf129ce2a1d/tfx/dependencies.py#L15-L36).

Some end-to-end tests in TFX uses MySQL as a database for `Apache Airflow`
orchestrator. You might need to install mysql client libraries in your
environment. For example, if you runs tests on Debian/Ubuntu, following command
will install required library: `sudo apt install libmysqlclient-dev`

If you have a local change in `.proto` files, you should re-generate the
protobuf stub code before using it with the following command. (This is
automatically invoked once when you first install `tfx` in editable mode, but
further stub generation requires manual invocation of the following command.)

```shell
# In the tfx root directory
bazel run //build:gen_proto
```

## Running Unit Tests

At this point all unit tests are safe to run externaly. We are working on
porting the end to end tests.

Each test can just be invoked with `python`. To invoke all unit tests:

```shell
find . -name '*_test.py' | grep -v e2e | xargs -I {} python {}
```

## Runing pylint

All new / changed code should pass [pylint](https://www.pylint.org/) linter.
Use pylint to check lint errors before sending a pull request. TFX has a
dedicated [pylintrc](https://github.com/tensorflow/tfx/blob/master/pylintrc).

```shell
pylint --rcfile <path_to_pylintrc> some_python.py
```

If your working directory is root of the tfx repository, you can omit `--rcfile`
flag. pylintrc in current directory will be used by default.

There are some existing issues especially with the lint tools. Googlers don't
use external tools like pylint, and the lint rules are a little bit
different. As a result, some existing code doesn't follow lint rule specified
in `pylintrc` configuration file. So keep in mind that there might be existing
issues, and you don't need to fix lint errors in existing code. But don't let
them grow.


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
