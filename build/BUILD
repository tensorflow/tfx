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

sh_binary(
    name = "gen_proto",
    srcs = ["gen_proto.sh"],
    data = [
        "//tfx/examples/custom_components/presto_example_gen/proto:presto_config_pb2.py",
        "//tfx/extensions/experimental/kfp_compatibility/proto:kfp_component_spec_pb2.py",
        "//tfx/extensions/google_cloud_big_query/experimental/elwc_example_gen/proto:elwc_config_pb2.py",
        "//tfx/orchestration/experimental/centralized_kubernetes_orchestrator/service/proto:service_pb2.py",
        "//tfx/orchestration/experimental/centralized_kubernetes_orchestrator/service/proto:service_pb2_grpc.py",
        "//tfx/orchestration/kubeflow/proto:kubeflow_pb2.py",
        "//tfx/proto:bulk_inferrer_pb2.py",
        "//tfx/proto:distribution_validator_pb2.py",
        "//tfx/proto:evaluator_pb2.py",
        "//tfx/proto:example_diff_pb2.py",
        "//tfx/proto:example_gen_pb2.py",
        "//tfx/proto:infra_validator_pb2.py",
        "//tfx/proto:pusher_pb2.py",
        "//tfx/proto:range_config_pb2.py",
        "//tfx/proto:trainer_pb2.py",
        "//tfx/proto:transform_pb2.py",
        "//tfx/proto:tuner_pb2.py",
        "//tfx/proto/orchestration:driver_output_pb2.py",
        "//tfx/proto/orchestration:executable_spec_pb2.py",
        "//tfx/proto/orchestration:execution_invocation_pb2.py",
        "//tfx/proto/orchestration:execution_result_pb2.py",
        "//tfx/proto/orchestration:execution_watcher_pb2.py",
        "//tfx/proto/orchestration:execution_watcher_pb2_grpc.py",
        "//tfx/proto/orchestration:garbage_collection_policy_pb2.py",
        "//tfx/proto/orchestration:local_deployment_config_pb2.py",
        "//tfx/proto/orchestration:metadata_pb2.py",
        "//tfx/proto/orchestration:pipeline_pb2.py",
        "//tfx/proto/orchestration:placeholder_pb2.py",
        "//tfx/proto/orchestration:platform_config_pb2.py",
        "//tfx/proto/orchestration:run_state_pb2.py",
        "//tfx/utils/testdata:bar_pb2.py",
        "//tfx/utils/testdata:foo_pb2.py",
    ],
)
