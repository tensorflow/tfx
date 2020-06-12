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
licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//tfx/orchestration/kubeflow/proto:kubeflow_pb2.py",
        "//tfx/proto:bulk_inferrer_pb2.py",
        "//tfx/proto:evaluator_pb2.py",
        "//tfx/proto:example_gen_pb2.py",
        "//tfx/proto:infra_validator_pb2.py",
        "//tfx/proto:pusher_pb2.py",
        "//tfx/proto:trainer_pb2.py",
        "//tfx/proto/orchestration:execution_result_pb2.py",
        "//tfx/proto/orchestration:executor_invocation_pb2.py",
        "//tfx/proto/orchestration:pipeline_pb2.py",
    ],
)
