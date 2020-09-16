workspace(name = "tfx")

# To update TensorFlow to a new revision.
# 1. Update the '_TENSORFLOW_GIT_COMMIT' var below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# LINT.IfChange(tf_commit)
# https://github.com/tensorflow/tensorflow/commits/v1.15.2
_TENSORFLOW_GIT_COMMIT = "5d80e1e8e6ee999be7db39461e0e79c90403a2e4"
# LINT.ThenChange(
#   :bazel_skylib,
#   :io_bazel_rules_clousure,
#   //tensorflow_metadata/WORKSPACE:tf_commit,
#   //tfx_bsl/WORKSPACE:tf_commit,
#   //tensorflow_data_validation/WORKSPACE:tf_commit,
#   //ml_metadata/WORKSPACE:tf_commit,
# )

http_archive(
    name = "org_tensorflow",
    sha256 = "7e3c893995c221276e17ddbd3a1ff177593d00fc57805da56dcc30fdc4299632",
    urls = [
      # Bazel mirror disabled due to b/162781348.
      # "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
)

# Needed by tf_py_wrap_cc rule from Tensorflow.
# When upgrading tensorflow version, also check tensorflow/WORKSPACE for the
# version of this -- keep in sync.
# LINT.IfChange(bazel_skylib)
# https://github.com/tensorflow/tensorflow/blob/v1.15.2/WORKSPACE
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)
# LINT.ThenChange(:tf_commit)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
# LINT.IfChange(io_bazel_rules_clousure)
# https://github.com/tensorflow/tensorflow/blob/v1.15.2/WORKSPACE
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2020-02-14
    ],
)
# LINT.ThenChange(:tf_commit)

# MLMD depends on "io_bazel_rules_go" so we need this here.
# LINT.IfChange(io_bazel_rules_go)
http_archive(
    name = "io_bazel_rules_go",
    sha256 = "492c3ac68ed9dcf527a07e6a1b2dcbf199c6bf8b35517951467ac32e421c06c1",
    urls = ["https://github.com/bazelbuild/rules_go/releases/download/0.17.0/rules_go-0.17.0.tar.gz"],
)
# LINT.ThenChange(
#   //ml_metadata/WORKSPACE:io_bazel_rules_go,
# )

# Please add all new TFX dependencies in workspace.bzl.
load("//tfx:workspace.bzl", "tfx_workspace")

tfx_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

# LINT.IfChange(bazel_version)
check_bazel_version_at_least("0.24.1")
# LINT.ThenChange(
#   //tensorflow_metadata/WORKSPACE:bazel_version,
#   //tfx_bsl/WORKSPACE:bazel_version,
#   //tensorflow_data_validation/WORKSPACE:bazel_version,
#   //ml_metadata/WORKSPACE:bazel_version,
# )
