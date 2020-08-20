# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.launcher."""
import os

import tensorflow as tf
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_executor_operator
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import test_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2

_PYTHON_CLASS_EXECUTABLE_SPEC = local_deployment_config_pb2.ExecutableSpec.PythonClassExecutableSpec


class FakeExecutorOperator(base_executor_operator.BaseExecutorOperator):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_SPEC_TYPE = None

  def run_executor(
      self, execution_info: base_executor_operator.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    return execution_result_pb2.ExecutorOutput()


class FakeCrashingExecutorOperator(base_executor_operator.BaseExecutorOperator):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_SPEC_TYPE = None

  def run_executor(
      self, execution_info: base_executor_operator.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    raise RuntimeError()


class LauncherTest(test_utils.TfxTest):

  def setUp(self):
    super(LauncherTest, self).setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)
    self._testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')

    # Sets up pipelines
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata',
            'pipeline_for_launcher_test.pbtxt'), pipeline)
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)
    self._pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        'test_run_0')

    # Extracts components
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node

    # Fakes an ExecutorSpec for Trainer
    self._trainer_executor_spec = _PYTHON_CLASS_EXECUTABLE_SPEC()
    # Fakes an executor operator
    self._test_executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: FakeExecutorOperator
    }

  @staticmethod
  def fakeUpstreamOutputs(mlmd_connection: metadata.Metadata,
                          example_gen: pipeline_pb2.PipelineNode,
                          transform: pipeline_pb2.PipelineNode):

    with mlmd_connection as m:
      if example_gen:
        # Publishes ExampleGen output.
        output_example = types.Artifact(
            example_gen.outputs.outputs['output_examples'].artifact_spec.type)
        output_example.uri = 'my_examples_uri'
        contexts = context_lib.register_contexts_if_not_exists(
            m, example_gen.contexts)
        execution = execution_publish_utils.register_execution(
            m, example_gen.node_info.type, contexts)
        execution_publish_utils.publish_succeeded_execution(
            m, execution.id, contexts, {
                'output_examples': [output_example],
            })

      if transform:
        # Publishes Transform output.
        output_transform_graph = types.Artifact(
            transform.outputs.outputs['transform_graph'].artifact_spec.type)
        output_example.uri = 'my_transform_graph_uri'
        contexts = context_lib.register_contexts_if_not_exists(
            m, transform.contexts)
        execution = execution_publish_utils.register_execution(
            m, transform.node_info.type, contexts)
        execution_publish_utils.publish_succeeded_execution(
            m, execution.id, contexts, {
                'transform_graph': [output_transform_graph],
            })

  def testLauncher_InputNotReady(self):
    # No new execution is triggered and registered if all inputs are not ready.
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators)
    execution_metadata = test_launcher.launch()

    self.assertIsNone(execution_metadata)
    with self._mlmd_connection as m:
      # No execution is registered in MLMD.
      self.assertEmpty(m.store.get_executions())

  def testLauncher_InputPartiallyReady(self):
    # No new execution is triggered and registered if all inputs are not ready.
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     None)
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators)

    with self._mlmd_connection as m:
      existing_exeuctions = m.store.get_executions()

    execution_metadata = test_launcher.launch()
    self.assertIsNone(execution_metadata)

    with self._mlmd_connection as m:
      # No new execution is registered in MLMD.
      self.assertCountEqual(existing_exeuctions, m.store.get_executions())

  def testLauncher_EmptyOptionalInputTriggersExecution(self):
    # In this test case, both inputs of trainer are mark as optional. So even
    # when there is no input from them, the trainer can stil be triggered.
    self._trainer.inputs.inputs['examples'].min_count = 0
    self._trainer.inputs.inputs['transform_graph'].min_count = 0
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators)
    execution_metadata = test_launcher.launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 5
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_0:my_trainer:model:0"
            }
          }
          state: LIVE""",
          artifact,
          ignored_fields=[
              'uri', 'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_metadata.id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          type_id: 3
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

  def testLauncher_PushingNewArtifactsAndUseCache(self):
    # In this test case, there are two executions:
    # In the first one,trainer reads the fake upstream outputs and publish
    # a new output.
    # In the second one, because the enable_cache is true and inputs don't
    # change. The launcher will published a CACHED execution.
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators)
    execution_metadata = test_launcher.launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 3
          type_id: 9
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_0:my_trainer:model:0"
            }
          }
          state: LIVE""",
          artifact,
          ignored_fields=[
              'uri', 'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_metadata.id])
      self.assertProtoPartiallyEquals(
          """
          id: 3
          type_id: 7
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

    execution_metadata = test_launcher.launch()
    with self._mlmd_connection as m:
      [execution] = m.store.get_executions_by_id([execution_metadata.id])
      self.assertProtoPartiallyEquals(
          """
          id: 4
          type_id: 7
          last_known_state: CACHED
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

  def testLauncher_CacheDisabled(self):
    # In this test case, there are two executions:
    # In the first one,trainer reads the fake upstream outputs and publish
    # a new output.
    # In the second one, because the enable_cache is false and inputs don't
    # change. The launcher will published a new COMPLETE execution.
    self._trainer.execution_options.caching_options.enable_cache = False

    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators)
    execution_metadata = test_launcher.launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 3
          type_id: 9
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_0:my_trainer:model:0"
            }
          }
          state: LIVE""",
          artifact,
          ignored_fields=[
              'uri', 'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_metadata.id])
      self.assertProtoPartiallyEquals(
          """
          id: 3
          type_id: 7
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

    execution_metadata = test_launcher.launch()
    with self._mlmd_connection as m:
      artifacts = m.store.get_artifacts_by_type('Model')
      self.assertLen(artifacts, 2)
      self.assertProtoPartiallyEquals(
          """
          id: 4
          type_id: 9
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_0:my_trainer:model:0"
            }
          }
          state: LIVE""",
          artifacts[1],
          ignored_fields=[
              'uri', 'create_time_since_epoch', 'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_metadata.id])
      self.assertProtoPartiallyEquals(
          """
          id: 4
          type_id: 7
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])

  def testLauncher_ExecutionFailed(self):
    # In the case that the executor failed and raises an execption.
    # An Execution will be published.
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: FakeCrashingExecutorOperator
    }
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=executor_operators)

    try:
      _ = test_launcher.launch()
    except:  # pylint: disable=bare-except
      pass

    with self._mlmd_connection as m:
      artifacts = m.store.get_artifacts_by_type('Model')
      self.assertEmpty(artifacts)
      executions = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 3
          type_id: 7
          last_known_state: FAILED
          """,
          executions[-1],
          ignored_fields=[
              'create_time_since_epoch', 'last_update_time_since_epoch'
          ])


if __name__ == '__main__':
  tf.test.main()
