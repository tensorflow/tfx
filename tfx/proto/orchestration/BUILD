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
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tfx:tfx.bzl", "tfx_py_proto_library")

tfx_py_proto_library(
    name = "execution_result_py_pb2",
    srcs = ["execution_result.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "execution_invocation_py_pb2",
    srcs = ["execution_invocation.proto"],
    deps = [
        ":pipeline_py_pb2",
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_service_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "pipeline_py_pb2",
    srcs = ["pipeline.proto"],
    deps = [
        ":placeholder_py_pb2",
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "executable_spec_py_pb2",
    srcs = ["executable_spec.proto"],
    deps = [":placeholder_py_pb2"],
)

tfx_py_proto_library(
    name = "local_deployment_config_py_pb2",
    srcs = ["local_deployment_config.proto"],
    deps = [
        ":executable_spec_py_pb2",
        ":platform_config_py_pb2",
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "driver_output_py_pb2",
    srcs = ["driver_output.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "placeholder_py_pb2",
    srcs = ["placeholder.proto"],
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "platform_config_py_pb2",
    srcs = ["platform_config.proto"],
)

tfx_py_proto_library(
    name = "execution_watcher_py_pb2",
    srcs = ["execution_watcher.proto"],
    use_grpc_plugin = True,
    deps = [
        "@com_github_google_ml_metadata//ml_metadata/proto:metadata_store_py_pb2",
    ],
)

tfx_py_proto_library(
    name = "run_state_py_pb2",
    srcs = ["run_state.proto"],
)
