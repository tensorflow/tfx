# TFX package

TFX is packaged as the `tfx` package on PyPI. We recommend that users install
TFX using `pip install tfx`. As of version 0.26.0, users also have the option
to install a standalone version of the TFX pipeline authoring SDK, as the
`ml-pipelines-sdk` package. This package has minimal dependencies, but does not
include first-party TFX components like the TFX ExampleGen, Transform and
Trainer in `tfx.components.*`, nor any additional tools requiring these
components.

Both the `tfx` and `ml-pipelines-sdk` packages share the `tfx` namespace and
the same source repository at https://github.com/tensorflow/tfx. These two
packages can be built using the instructions below. During development, a
single editable package may be installed for convenience (see the "Installing
the development-only `tfx-dev` package" section below).

# Building TFX pip package from source

## Setting up the build environment

First, set up the build environment by running:

```
package_build/initialize.sh
```

## Building the `tfx` and `ml-pipelines-sdk` packages

Next, each package can be built using the `bdist_wheel` command:

```
python package_build/ml-pipelines-sdk/setup.py bdist_wheel
python package_build/tfx/setup.py bdist_wheel
```

As a result, `.whl` files will be generated in the `dist/` directory.

# Installing the development-only `tfx-dev` package

During development, it is convenient to install a single editable pip package.
This package will contain the union of the `tfx` and `ml-pipelines-sdk`
package in an editable environment. To install this combined package for
development, run from the repository root:

```
pip install -e .
```

This `tfx-dev` package should not be packaged as a binary or source
distribution using `python setup.py {bdist_wheel,sdist}` to avoid conflicts
with the two official `tfx` and `ml-pipelines-sdk` packages. Instead, users
should build the two packages for distribution with the directions above.

# Temporary workaround for building `tfx-dev` wheels.

To minimize dependency issues, the instructions above should be used to build
TFX wheel files for deployment. As a temporary workaround, the environmental
variable `UNSUPPORTED_BUILD_TFX_DEV_WHEEL` may be set to `1` to forcibly enable
building and installation of a single `tfx-dev` pip package containing the union
of the `tfx` and `ml-pipelines-sdk` packages. This workaround may lead to
package namespace conflicts and is not recommended or supported, and will be
removed in a future version.

