# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Util for compiling NodeExecutionOptions dataclasss into NodeExecutionOptionsProto."""


from tfx.dsl.experimental.node_execution_options import utils as node_execution_options_utils
from tfx.proto.orchestration import pipeline_pb2


def compile_node_execution_options(
    options_py: node_execution_options_utils.NodeExecutionOptions,
) -> pipeline_pb2.NodeExecutionOptions:
  """Compiles NodeExecutionOptions dataclass into NodeExecutionOptionsProto."""
  options_pb = pipeline_pb2.NodeExecutionOptions()
  options_pb.strategy = options_py.trigger_strategy
  options_pb.node_success_optional = options_py.success_optional
  if options_py.max_execution_retries is not None:
    options_pb.max_execution_retries = options_py.max_execution_retries
  options_pb.execution_timeout_sec = options_py.execution_timeout_sec
  options_pb.run_mode = options_py._run_mode  # pylint: disable=protected-access
  options_pb.reset_stateful_working_dir = options_py.reset_stateful_working_dir
  if options_py.lifetime_start:
    options_pb.resource_lifetime.lifetime_start = options_py.lifetime_start

  return options_pb
