workspace(name = "tfx")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_google_protobuf",
    sha256 = "597071a340acc5346494c119ba3a541825c3f81071fc783521b24e29a485d60f",
    strip_prefix = "protobuf-6.31.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.tar.gz"],
    patch_args = ["-p1", "-l"],
    patches = ["//patches:com_google_protobuf_compat.patch"],
    repo_mapping = {
        "@abseil-cpp": "@com_google_absl",
    },
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

http_archive(
    name = "rules_java",
    urls = [
        "https://github.com/bazelbuild/rules_java/releases/download/8.7.0/rules_java-8.7.0.tar.gz",
    ],
    sha256 = "5449ed36d61269579dd9f4b0e532cd131840f285b389b3795ae8b4d717387dd8",
)

load("@rules_java//java:rules_java_deps.bzl", "rules_java_dependencies")
rules_java_dependencies()

load("@rules_java//java:repositories.bzl", "rules_java_toolchains")
rules_java_toolchains()

http_archive(
    name = "rules_cc",
    sha256 = "abc605dd850f813bb37004b77db20106a19311a96b2da1c92b789da529d28fe1",
    strip_prefix = "rules_cc-0.0.17",
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.17/rules_cc-0.0.17.tar.gz"],
)

# TF 2.21
# LINT.IfChange(tf_commit)
_TENSORFLOW_GIT_COMMIT = "a481b10260dfdf833a1b16007eead49c1d7febf3"
# LINT.ThenChange(:io_bazel_rules_clousure)
http_archive(
    name = "org_tensorflow",
    sha256 = "6438396f3b19af5d7ad787cf041f857af7505916dc08092e20b07d1b1f8df492",
    urls = [
      # Bazel mirror disabled due to b/162781348.
      # "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

# Needed by tf_py_wrap_cc rule from Tensorflow.
# When upgrading tensorflow version, also check tensorflow/WORKSPACE for the
# version of this -- keep in sync.
http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
# LINT.IfChange(io_bazel_rules_clousure)
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

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "b4df908ec14868369021182ab191dbd1f40830c9b300650d5dc389e0b9266c8d",
    url = "https://github.com/bazelbuild/rules_apple/releases/download/3.5.1/rules_apple.3.5.1.tar.gz",
)

# Needed by gRPC.
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "1ae6fcf983cff3edab717636f91ad0efff2e5ba75607fdddddfd6ad0dbdfaf10",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/1.24.5/apple_support.1.24.5.tar.gz"],
)

http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "bb01097c7c7a1407f8ad49a1a0b1960655cf823c26ad2782d0b7d15b323838e2",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/1.18.0/rules_swift.1.18.0.tar.gz"],
)

# Initialize hermetic Python
load("@org_tensorflow//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["@org_tensorflow//:WORKSPACE"],
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
    },
)

load("@org_tensorflow//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@org_tensorflow//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()



load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

# Please add all new TFX dependencies in workspace.bzl.
load("//tfx:workspace.bzl", "tfx_workspace")
tfx_workspace()

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("5.3.0")
