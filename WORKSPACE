# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

workspace(name = "tfx")

# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.

load("//tfx:repo.bzl", "tensorflow_http_archive")

# v1.13.0-rc0
tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "1677e8e5bf9d5a9041aa5fc12ea6ff82eecc8603f6fd1292f0759b90e7096c21",
    git_commit = "a8e5c41c5bbe684a88b9285e07bd9838c089e83b",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "43c9b882fa921923bcba764453f4058d102bece35a37c9f6383c713004aacff1",
    strip_prefix = "rules_closure-9889e2348259a5aad7e805547c1a0cf311cfcd91",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/9889e2348259a5aad7e805547c1a0cf311cfcd91.tar.gz",  # 2018-12-21
    ],
)

# Please add all new ML Metadata dependencies in workspace.bzl.
load("//tfx:workspace.bzl", "tfx_workspace")
tfx_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least(minimum_bazel_version="0.18.0")
