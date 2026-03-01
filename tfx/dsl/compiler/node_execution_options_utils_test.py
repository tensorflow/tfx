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
"""Tests for tfx.dsl.compiler.node_execution_options_utils."""


from absl.testing import absltest
from tfx.dsl.compiler import node_execution_options_utils
from tfx.dsl.experimental.node_execution_options import utils as neo_lib
from tfx.proto.orchestration import pipeline_pb2


class NodeExecutionOptionsUtilsTest(absltest.TestCase):

  def test_compiles(self):
    options_py = neo_lib.NodeExecutionOptions(
        trigger_strategy=neo_lib.TriggerStrategy.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED,
        success_optional=True,
        execution_timeout_sec=12345,
        reset_stateful_working_dir=True,
    )
    self.assertEqual(
        node_execution_options_utils.compile_node_execution_options(options_py),
        pipeline_pb2.NodeExecutionOptions(
            strategy=pipeline_pb2.NodeExecutionOptions.TriggerStrategy.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED,
            node_success_optional=True,
            execution_timeout_sec=12345,
            reset_stateful_working_dir=True,
        ),
    )

  def test_compiles_retries(self):
    options_py = neo_lib.NodeExecutionOptions(max_execution_retries=4)
    self.assertEqual(
        node_execution_options_utils.compile_node_execution_options(options_py),
        pipeline_pb2.NodeExecutionOptions(
            strategy=pipeline_pb2.NodeExecutionOptions.TriggerStrategy.ALL_UPSTREAM_NODES_SUCCEEDED,
            max_execution_retries=4,
        ),
    )

  def test_compiles_lifetime_start(self):
    options_py = neo_lib.NodeExecutionOptions(lifetime_start='start-node')
    self.assertEqual(
        node_execution_options_utils.compile_node_execution_options(options_py),
        pipeline_pb2.NodeExecutionOptions(
            strategy=pipeline_pb2.NodeExecutionOptions.TriggerStrategy.ALL_UPSTREAM_NODES_SUCCEEDED,
            resource_lifetime=pipeline_pb2.NodeExecutionOptions.ResourceLifetime(
                lifetime_start='start-node'
            ),
        ),
    )
