# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
package(default_visibility = ["//tfx:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tfx:tfx.bzl", "tfx_py_proto_library")

tfx_py_proto_library(
    name = "bulk_inferrer_proto_py_pb2",
    srcs = ["bulk_inferrer.proto"],
)

tfx_py_proto_library(
    name = "evaluator_proto_py_pb2",
    srcs = ["evaluator.proto"],
)

tfx_py_proto_library(
    name = "example_gen_proto_py_pb2",
    srcs = ["example_gen.proto"],
)

tfx_py_proto_library(
    name = "infra_validator_py_pb2",
    srcs = ["infra_validator.proto"],
)

tfx_py_proto_library(
    name = "pusher_proto_py_pb2",
    srcs = ["pusher.proto"],
)

tfx_py_proto_library(
    name = "range_config_py_pb2",
    srcs = ["range_config.proto"],
)

tfx_py_proto_library(
    name = "trainer_proto_py_pb2",
    srcs = ["trainer.proto"],
)

tfx_py_proto_library(
    name = "transform_proto_py_pb2",
    srcs = ["transform.proto"],
)

tfx_py_proto_library(
    name = "tuner_proto_py_pb2",
    srcs = ["tuner.proto"],
)

tfx_py_proto_library(
    name = "example_diff_proto_py_pb2",
    srcs = ["example_diff.proto"],
)

tfx_py_proto_library(
    name = "distribution_validator_proto_py_pb2",
    srcs = ["distribution_validator.proto"],
    deps = [
        "@com_github_tf_metadata//tensorflow_metadata/proto/v0:py_metadata_v0_proto_py",
    ],
)
