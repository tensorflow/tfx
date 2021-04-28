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
"""Proto library helper utils.
"""

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

def tfx_py_proto_library(name, srcs = [], deps = [], visibility = None, testonly = 0, use_grpc_plugin = False):
    """Opensource py_proto_library."""
    py_proto_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        deps = ["@com_google_protobuf//:protobuf_python"] + deps,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        visibility = visibility,
        testonly = testonly,
        use_grpc_plugin = use_grpc_plugin,
    )
