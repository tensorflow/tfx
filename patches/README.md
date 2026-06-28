# TFX Patches

This directory contains patch files needed to work around compatibility issues in the TFX build system.

## Patch Files

### 1. `tfx_bsl.patch`

**Purpose:** Fixes ZetaSQL dependency download in the tensorflow/tfx-bsl repository during Docker image builds.

**Details:**
- Applied during the Docker image build process (defined in `tfx/tools/docker/Dockerfile`)
- Updates the ZetaSQL URL from `zetasql` to `googlesql` in the repository name
- Updates the corresponding SHA256 hash
- Enables correct fetching of ZetaSQL when building the tfx-bsl dependency

**Removal Condition:**
- The upstream tfx-bsl repository should update its WORKSPACE to use the corrected ZetaSQL repository and hash
- Moving the entire TFX ecosystem to GitHub Actions-based builds and upgrading the base Docker image would eliminate the need for from-source builds inside the Docker container, making it easier to remove this patch
- This is part of improving the overall dependency resolution process

---

### 2. `tfdv.patch`

**Purpose:** Fixes ZetaSQL dependency download in the tensorflow/data-validation repository during Docker image builds.

**Details:**
- Applied during the Docker image build process (defined in `tfx/tools/docker/Dockerfile`)
- Updates the ZetaSQL URL from `zetasql` to `googlesql` in the repository name
- Updates the corresponding SHA256 hash
- Pins specific git commits for `tensorflow/metadata` and `tensorflow/tfx-bsl` branches instead of using `master` branch references
  - `tensorflow-metadata`: pinned to commit `51d688ffc0e1b94e6298981df58eecb4ef47ab9a`
  - `tfx-bsl`: pinned to commit `ed8aeec4f00680f9e88cd75c85ee60d8aa6789df`
- Ensures reproducible builds and works around transient issues from following moving target branches

**Removal Condition:**
- The upstream tensorflow/data-validation repository should update its WORKSPACE to use the corrected ZetaSQL repository and hash
- Moving the entire TFX ecosystem to GitHub Actions-based builds and upgrading the base Docker image would eliminate the need for from-source builds inside the Docker container, making it easier to remove this patch

---

### 3. `tfx.patch`

**Purpose:** Removes TFX dependencies that cannot be reliably installed via pip from PyPI with compatible versions during Docker image builds.

**Details:**
- Applied in `build_docker_image.sh` during Docker image build initialization
- Removes the following dependencies from constraint files and dependencies.py:
  - `tensorflow-cloud`
  - `tensorflow-data-validation`
  - `tensorflow-transform`
  - `tfx-bsl`
- These dependencies are instead manually installed via pre-downloaded `.whl` files (wheels are downloaded separately and added to the Docker build context)
- The patch is automatically reverted after successful Docker build completion
- If the Docker build fails, the patch must be manually reverted with: `git apply -R patches/tfx.patch`

**Removal Conditions:**
1. **GitHub Actions-based Release Process:** TFX should transition to a GitHub Actions-based release workflow that builds all wheels in a consistent, unified platform environment. This ensures that all `.whl` files are built reproducibly and are available on PyPI with compatible versions for the current TensorFlow version.

2. **Base Docker Image Upgrade:** The base Docker image used in the build should be upgraded to a version where all `.whl` files available on PyPI are directly installable without conflicts or platform-specific compatibility issues.

Once these conditions are met, the dependencies can be listed in constraint files and `dependencies.py` directly, and the patch can be removed.

---

### 4. `tensorflow_metadata_proto_v0.patch`

**Purpose:** Handles protocol buffer compatibility with Protobuf 21.12 in the tensorflow/metadata repository.

**Details:**
- Not related to Docker image builds; applies to native builds
- Addresses Protobuf API changes between versions
- Key changes:
  - Removes the `load` statement for `py_proto_library` from `@com_google_protobuf//bazel:py_proto_library.bzl`
  - Removes unnecessary Protobuf well-known type dependencies that are not actually used in metadata proto definitions:
    - `api_proto`
    - `compiler_plugin_proto`
    - `empty_proto`
    - `field_mask_proto`
    - `source_context_proto`
    - `type_proto`
  - Converts `py_proto_library` targets to simpler `alias` targets pointing to the generated proto libraries

**Removal Condition:**
- Upgrade `tensorflow-metadata` to use **Protobuf 4.25.6** or later
- This requires upstream work in the tensorflow/metadata repository to update its protobuf definitions and BUILD files to be compatible with newer Protobuf versions
- Once metadata is compatible with Protobuf 4.25.6, this patch can be removed
