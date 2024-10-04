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

At this point, TFX only supports Python 3 (3.7, 3.8, and 3.9) on Linux and
MacOS. Please use one of these operating systems for development and testing.

## Setting up local development environment

### Installing Bazel

To test local change, first you have to install
[Bazel](https://bazel.build/install), which powers the protobuf stub code
generation. Check whether Bazel is installed and executable:

```shell
bazel --version
```

After installing Bazel, you can move to the cloned source directory.

```shell
pushd <your_source_dir>
```

### Installing TFX

TFX has many dependent family libraries like TensorFlow Data Validation and
TensorFlow Model Analysis. Sometimes, TFX uses their most recent API changes
before published. So it is safer to use nightly versions of those libraries when
you develop TFX. You have to set property dependency using
`TFX_DEPENDENCY_SELECTOR` environment variable, and supply our nightly package
index URL when installing TFX.

> NOTE: Please use the latest version of `pip` (Possibly after 21.0) before
> running following commands. TFX works well with the new dependency resolver
> from TFX 0.30.0. `pip install --upgrade pip` should upgrade pip.

You can install TFX source code in a virtual environment in editable (`-e`)
mode, which will pick up your local changes immediately without re-installing
every time.

> NOTE: If you are making changes to TFX that also require changes in a
> TFX dependency, such as TFX-BSL, please issue separate PRs for each repo in
> an order which avoids breaking tests.

```shell
export TFX_DEPENDENCY_SELECTOR=NIGHTLY
pip install -e . --extra-index-url https://pypi-nightly.tensorflow.org/simple
```

You may need to pass `--use-deprecated=legacy-resolver` if the pip resolver is
taking too long.

You might need to install additional packages to run all end-to-end tests. To
run all tests, use `[test]` extra requirements which includes all dependencies
including airflow and kfp. If you want to test a specific orchestrator only, use
`[airflow]` or `[kfp]`. (Beam and Local orchestrators can be run without any
extra dependency.) For example,

```shell
pip install -e .[kfp] --extra-index-url https://pypi-nightly.tensorflow.org/simple
```

Alternatively, you can also build all TFX family libraries from GitHub source
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

### Re-generating protobuf stub code

If you have a local change in `.proto` files, you should re-generate the
protobuf stub code before using it with the following command. (This is
automatically invoked once when you first install `tfx` in editable mode, but
further stub generation requires manual invocation of the following command.)

```shell
# In the tfx root directory
bazel run //build:gen_proto
```

## Testing

### Testing Conventions

All python unit tests in this repo is based on Tensorflow's
[tf.test.TestCase](https://www.tensorflow.org/api_docs/python/tf/test/TestCase),
which is a subclass of
[py-absl TestCase](https://github.com/abseil/abseil-py/blob/06edd9c20592cec39178b94240b5e86f32e19768/absl/testing/absltest.py#L523).

We have several types of tests in this repo:

*   Unit tests for source code;
*   End to end tests (filenames end with `_e2e_test.py`): some of these also run
    with external environments;
*   Integration tests (filenames end with `_integration_test.py`): some of these might
    run with external environments;
*   Performance tests (filenames end with `_perf_test.py`): some of these might
    run with external environments.

### Running Unit Tests

At this point all unit tests are safe to run externally. We are working on
porting the end to end tests.

To run all tests:

```shell
pytest
```

Each test can be run individually with `pytest`:

```shell
pytest tfx/a_module/a_particular_test.py
```

Some tests are slow and are given the `pytest.mark.slow` mark. These tests
are slow and/or require more dependencies.

```shell
pytest -m "slow"
```

To invoke all unit tests not marked as slow:

```shell
pytest -m "not slow"
```

To invoke end to end tests:

```shell
pytest -m "e2e"
```

To skip end to end tests:

```shell
pytest -m "not e2e"
```

To invoke integration tests:

```shell
pytest -m "integration"
```

To invoke performance tests:

```shell
pytest -m "perf"
```

## Running pylint

We follow
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) in
our code. All new / changed code should pass [pylint](https://pylint.pycqa.org/)
linter using the dedicated [pylintrc](./pylintrc). Use pylint to check lint
errors before sending a pull request.

```shell
pylint --rcfile ./pylintrc some_python.py
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

## Continuous Integration

This project makes use of CI for

- Building the `tfx` python package when releases are made
- Running tests
- Linting pull requests
- Building documentation

These four _workflows_ trigger automatically when certain _events_ happen.

### Pull Requests

When a PR is made:

- Wheels and an sdist are built using the code in the PR branch. Multiple wheels
  are built for a [variety of architectures and python
  versions](https://github.com/tensorflow/tfx/blob/master/.github/workflows/wheels.yml).
  If the PR causes any of the wheels to fail to build, the failure will be
  reported in the checks for the PR.

- Tests are run via [`pytest`](https://github.com/tensorflow/tfx/blob/master/.github/workflows/ci-test.yml). If a test fails, the workflow failure will be
  reported in the checks for the PR.

- Lint checks are run on the changed files. This workflow makes use of the
  [`.pre-commit-config.yaml`](https://github.com/tensorflow/tfx/blob/master/.pre-commit-config.yaml), and if any lint violations are found the workflow
  reports a failure on the list of checks for the PR.

If the author of the PR makes a new commit to the PR branch, these checks are
run again on the new commit.

### Releases

When a release is made on GitHub the workflow that builds wheels runs, just as
it does for pull requests, but with one difference: it automatically uploads the
wheels and sdist that are built in the workflow to the Python Package Index
(PyPI) using [trusted
publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/#configuring-trusted-publishing)
without any additional action required on the part of the release captain. After
the workflow finishes, users are able to use `pip install tfx` to install the
newly published version.

### Commits to `master`

When a new commit is made to the `master`, the documentation is built and
automatically uploaded to github pages.

If you want to see the changes to the documentation when rendered, run `mkdocs
serve` to build the documentation and serve it locally. Alternatively, if you
merge your own changes to your own fork's `master` branch, this workflow will
serve the documentation at `https://<your-github-username>.github.io/tfx`. This
provides a convenient way for developers to check deployments before they merge
a PR to the upstream `tfx` repository.
