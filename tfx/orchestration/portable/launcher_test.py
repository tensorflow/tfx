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
import contextlib
import copy
import os
from unittest import mock

import tensorflow as tf
from tfx import types
from tfx import version as tfx_version
from tfx.dsl.compiler import constants
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_driver
from tfx.orchestration.portable import base_executor_operator
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable import system_node_handler
from tfx.orchestration.portable.mlmd import context_lib
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2

_PYTHON_CLASS_EXECUTABLE_SPEC = executable_spec_pb2.PythonClassExecutableSpec


class FakeError(Exception):
  pass


class _FakeExecutorOperator(base_executor_operator.BaseExecutorOperator):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_CONFIG_TYPE = None

  def run_executor(
      self, execution_info: data_types.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    self._exec_properties = execution_info.exec_properties
    # The Launcher expects the ExecutorOperator to return an ExecutorOutput
    # structure with the output artifact information for MLMD publishing.
    output_dict = copy.deepcopy(execution_info.output_dict)
    result = execution_result_pb2.ExecutorOutput()
    for key, artifact_list in output_dict.items():
      artifacts = execution_result_pb2.ExecutorOutput.ArtifactList()
      for artifact in artifact_list:
        artifacts.artifacts.append(artifact.mlmd_artifact)
      result.output_artifacts[key].CopyFrom(artifacts)
    return result


class _FakeEmptyExecutorOperator(base_executor_operator.BaseExecutorOperator):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_CONFIG_TYPE = None

  def run_executor(
      self, execution_info: data_types.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    self._exec_properties = execution_info.exec_properties
    return execution_result_pb2.ExecutorOutput()


class _FakeCrashingExecutorOperator(base_executor_operator.BaseExecutorOperator
                                   ):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_CONFIG_TYPE = None

  def run_executor(
      self, execution_info: data_types.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    raise FakeError()


class _FakeErrorExecutorOperator(base_executor_operator.BaseExecutorOperator):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_CONFIG_TYPE = None

  def run_executor(
      self, execution_info: data_types.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    result = execution_result_pb2.ExecutorOutput()
    result.execution_result.code = 1
    result.execution_result.result_message = 'execution canceled.'
    return result


class _FakeCleanUpExecutorOperator(base_executor_operator.BaseExecutorOperator):

  SUPPORTED_EXECUTOR_SPEC_TYPE = [_PYTHON_CLASS_EXECUTABLE_SPEC]
  SUPPORTED_PLATFORM_CONFIG_TYPE = None

  def run_executor(
      self, execution_info: data_types.ExecutionInfo
  ) -> execution_result_pb2.ExecutorOutput:
    output_dict = copy.deepcopy(execution_info.output_dict)
    result = execution_result_pb2.ExecutorOutput()
    for key, artifact_list in output_dict.items():
      artifacts = execution_result_pb2.ExecutorOutput.ArtifactList()
      for artifact in artifact_list:
        artifacts.artifacts.append(artifact.mlmd_artifact)
      result.output_artifacts[key].CopyFrom(artifacts)
    # Although the following removing is typically not expected, but there is
    # no way to prevent them from happening. We should make sure that the
    # launcher can handle the double cleanup gracefully.
    fileio.rmtree(
        os.path.abspath(
            os.path.join(execution_info.stateful_working_dir, os.pardir)))
    fileio.rmtree(execution_info.tmp_dir)
    return result


class _FakeExampleGenLikeDriver(base_driver.BaseDriver):

  def __init__(self, mlmd_connection: metadata.Metadata):
    super().__init__(mlmd_connection)
    self._self_output = text_format.Parse(
        """
      inputs {
        key: "examples"
        value {
          channels {
            producer_node_query {
              id: "my_example_gen"
            }
            context_queries {
              type {
                name: "pipeline"
              }
              name {
                field_value {
                  string_value: "my_pipeline"
                }
              }
            }
            context_queries {
              type {
                name: "component"
              }
              name {
                field_value {
                  string_value: "my_example_gen"
                }
              }
            }
            artifact_query {
              type {
                name: "Examples"
              }
            }
            output_key: "output_examples"
          }
          min_count: 1
        }
      }""", pipeline_pb2.NodeInputs())

  def run(self, execution_info) -> driver_output_pb2.DriverOutput:
    # Fake a constant span number, which, on prod, is usually calculated based
    # on date.
    span = 2
    with self._mlmd_connection as m:
      previous_output = inputs_utils.resolve_input_artifacts(
          m, self._self_output)

      # Version should be the max of existing version + 1 if span exists,
      # otherwise 0.
      version = 0
      if previous_output:
        version = max([
            artifact.get_int_custom_property('version')
            for artifact in previous_output['examples']
            if artifact.get_int_custom_property('span') == span
        ] or [-1]) + 1

    output_example = copy.deepcopy(
        execution_info.output_dict['output_examples'][0].mlmd_artifact)
    output_example.custom_properties['span'].int_value = span
    output_example.custom_properties['version'].int_value = version
    result = driver_output_pb2.DriverOutput()
    result.output_artifacts['output_examples'].artifacts.append(output_example)

    result.exec_properties['span'].int_value = span
    result.exec_properties['version'].int_value = version
    return result


class LauncherTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Set a constant version for artifact version tag.
    patcher = mock.patch('tfx.version.__version__')
    patcher.start()
    tfx_version.__version__ = '0.123.4.dev'
    self.addCleanup(patcher.stop)

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    # Sets up pipelines
    self._pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata',
            'pipeline_for_launcher_test.pbtxt'), self._pipeline)
    self._pipeline_info = self._pipeline.pipeline_info
    self._pipeline_runtime_spec = self._pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)
    self._pipeline_run_id_counter = 0
    self.reloadPipelineWithNewRunId()

    # Fakes an ExecutorSpec for Trainer
    self._trainer_executor_spec = _PYTHON_CLASS_EXECUTABLE_SPEC()
    # Fakes an executor operator
    self._test_executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeExecutorOperator
    }
    # Fakes an custom driver spec
    self._custom_driver_spec = _PYTHON_CLASS_EXECUTABLE_SPEC()
    self._custom_driver_spec.class_path = 'tfx.orchestration.portable.launcher_test._FakeExampleGenLikeDriver'

  def reloadPipelineWithNewRunId(self):
    self._pipeline_run_id_counter += 1
    # Substitute the runtime parameter to be a new run_id.
    pipeline = pipeline_pb2.Pipeline()
    pipeline.CopyFrom(self._pipeline)
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME:
                f'test_run_{self._pipeline_run_id_counter}',
        })
    self._pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        f'test_run_{self._pipeline_run_id_counter}')

    # Extracts components
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node
    self._importer = pipeline.nodes[3].pipeline_node
    self._resolver = pipeline.nodes[4].pipeline_node

  @staticmethod
  def fakeUpstreamOutputs(mlmd_connection: metadata.Metadata,
                          example_gen: pipeline_pb2.PipelineNode,
                          transform: pipeline_pb2.PipelineNode,
                          cached: bool = False):
    publish_execution = (
        execution_publish_utils.publish_cached_execution
        if cached else execution_publish_utils.publish_succeeded_execution)
    with mlmd_connection as m:
      if example_gen:
        # Publishes ExampleGen output.
        output_example = types.Artifact(
            example_gen.outputs.outputs['output_examples'].artifact_spec.type)
        output_example.uri = 'my_examples_uri'
        contexts = context_lib.prepare_contexts(m, example_gen.contexts)
        execution = execution_publish_utils.register_execution(
            m, example_gen.node_info.type, contexts)
        publish_execution(
            metadata_handler=m,
            execution_id=execution.id,
            contexts=contexts,
            output_artifacts={
                'output_examples': [output_example],
            })

      if transform:
        # Publishes Transform output.
        output_transform_graph = types.Artifact(
            transform.outputs.outputs['transform_graph'].artifact_spec.type)
        output_transform_graph.uri = 'my_transform_graph_uri'
        contexts = context_lib.prepare_contexts(m, transform.contexts)
        execution = execution_publish_utils.register_execution(
            m, transform.node_info.type, contexts)
        publish_execution(
            metadata_handler=m,
            execution_id=execution.id,
            contexts=contexts,
            output_artifacts={
                'transform_graph': [output_transform_graph],
            })

  @staticmethod
  def fakeExampleGenOutput(mlmd_connection: metadata.Metadata,
                           example_gen: pipeline_pb2.PipelineNode, span: int,
                           version: int):
    with mlmd_connection as m:
      output_example = types.Artifact(
          example_gen.outputs.outputs['output_examples'].artifact_spec.type)
      output_example.set_int_custom_property('span', span)
      output_example.set_int_custom_property('version', version)
      output_example.uri = 'my_examples_uri'
      contexts = context_lib.prepare_contexts(m, example_gen.contexts)
      execution = execution_publish_utils.register_execution(
          m, example_gen.node_info.type, contexts)
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {
              'output_examples': [output_example],
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
    test_launcher.launch()

    with self._mlmd_connection as m:
      # One failed execution in MLMD.
      executions = m.store.get_executions()
      self.assertLen(executions, 1)
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: FAILED
          """,
          executions[0],
          ignored_fields=['type_id', 'custom_properties',
                          'create_time_since_epoch',
                          'last_update_time_since_epoch'])

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
      existing_execution_ids = {e.id for e in m.store.get_executions()}

    test_launcher.launch()

    with self._mlmd_connection as m:
      # One failed execution in MLMD.
      new_executions = [e for e in m.store.get_executions()
                        if e.id not in existing_execution_ids]
      self.assertLen(new_executions, 1)
      self.assertProtoPartiallyEquals(
          """
          id: 2
          last_known_state: FAILED
          """,
          new_executions[0],
          ignored_fields=['type_id', 'custom_properties',
                          'create_time_since_epoch',
                          'last_update_time_since_epoch'])

  def testLauncher_EmptyOptionalInputTriggersExecution(self):
    self.reloadPipelineWithNewRunId()
    self._test_executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeEmptyExecutorOperator
    }
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
    execution_info = test_launcher.launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 1
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_trainer:model:0"
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifact,
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 1
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_PublishingNewArtifactsAndUseCache(self):
    # In this test case, there are two executions:
    # In the first one,trainer reads the fake upstream outputs and publish
    # a new output.
    # In the second one, because the enable_cache is true and inputs don't
    # change. The launcher will published a CACHED execution.
    self.reloadPipelineWithNewRunId()
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    execution_info = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators).launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 3
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_trainer:model:0"
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifact,
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 3
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

    self.reloadPipelineWithNewRunId()
    execution_info = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators).launch()

    with self._mlmd_connection as m:
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 4
          last_known_state: CACHED
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_CacheIsSupportedForNodeWithNoOutput(self):
    # Even though a node has no output at all, the launcher should treat the
    # second execution as CACHED as long as the cache context is the same.
    self.reloadPipelineWithNewRunId()
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    self._trainer.ClearField('outputs')
    execution_info = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators).launch()

    with self._mlmd_connection as m:
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 3
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

    self.reloadPipelineWithNewRunId()
    self._trainer.ClearField('outputs')
    execution_info = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators).launch()

    with self._mlmd_connection as m:
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 4
          last_known_state: CACHED
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_CacheDisabled(self):
    # In this test case, there are two executions:
    # In the first one,trainer reads the fake upstream outputs and publish
    # a new output.
    # In the second one, because the enable_cache is false and inputs don't
    # change. The launcher will published a new COMPLETE execution.
    self.reloadPipelineWithNewRunId()
    self._trainer.execution_options.caching_options.enable_cache = False

    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    execution_info = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators).launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 3
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_trainer:model:0"
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifact,
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 3
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

    self.reloadPipelineWithNewRunId()
    self._trainer.execution_options.caching_options.enable_cache = False
    execution_info = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=self._test_executor_operators).launch()

    with self._mlmd_connection as m:
      artifacts = m.store.get_artifacts_by_type('Model')
      self.assertLen(artifacts, 2)
      self.assertProtoPartiallyEquals(
          """
          id: 4
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_trainer:model:0"
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifacts[1],
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 4
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_ReEntry(self):
    # Some executors or runtime environment may reschedule the launcher job
    # before the launcher job can publish any results of the execution to MLMD.
    # The launcher should reuse the previous execution and proceed to a
    # successful execution.
    self.reloadPipelineWithNewRunId()
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeCrashingExecutorOperator
    }
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=executor_operators)

    # The first launch simulates the launcher being restarted by preventing the
    # publishing of any results to MLMD.
    with contextlib.ExitStack() as stack:
      stack.enter_context(
          mock.patch.object(test_launcher, '_publish_failed_execution'))
      stack.enter_context(
          mock.patch.object(test_launcher,
                            '_clean_up_stateless_execution_info'))
      stack.enter_context(self.assertRaises(FakeError))
      test_launcher.launch()

    # Retrieve execution from the first launch, which should be in RUNNING
    # state.
    with self._mlmd_connection as m:
      contexts = context_lib.prepare_contexts(
          metadata_handler=m,
          node_contexts=test_launcher._pipeline_node.contexts)
      exec_properties = data_types_utils.build_parsed_value_dict(
          inputs_utils.resolve_parameters_with_schema(
              node_parameters=test_launcher._pipeline_node.parameters))
      input_artifacts = inputs_utils.resolve_input_artifacts(
          metadata_handler=m, node_inputs=test_launcher._pipeline_node.inputs)
      first_execution = test_launcher._register_or_reuse_execution(
          metadata_handler=m,
          contexts=contexts,
          input_artifacts=input_artifacts,
          exec_properties=exec_properties)
      self.assertEqual(first_execution.last_known_state,
                       metadata_store_pb2.Execution.RUNNING)

    # Create a second test launcher. It should reuse the previous execution.
    executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeExecutorOperator
    }
    second_test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=executor_operators)
    execution_info = second_test_launcher.launch()

    with self._mlmd_connection as m:
      self.assertEqual(first_execution.id, execution_info.execution_id)
      executions = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertLen(executions, 1)
      self.assertEqual(executions.pop().last_known_state,
                       metadata_store_pb2.Execution.COMPLETE)

  def testLauncher_ToleratesDoubleCleanup(self):
    # Some executors or runtime environment may delete stateful_working_dir,
    # tmp_dir and unexpectedly. The launcher should handle such cases gracefully
    # and proceed to a successful execution.
    self.reloadPipelineWithNewRunId()
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)

    executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeCleanUpExecutorOperator
    }
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=executor_operators)
    execution_info = test_launcher.launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Model')
      self.assertProtoPartiallyEquals(
          """
          id: 3
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_trainer:model:0"
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifact,
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])
      [execution] = m.store.get_executions_by_id([execution_info.execution_id])
      self.assertProtoPartiallyEquals(
          """
          id: 3
          last_known_state: COMPLETE
          """,
          execution,
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_ExecutionFailed(self):
    # In the case that the executor failed and raises an execption.
    # An Execution will be published.
    self.reloadPipelineWithNewRunId()
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeCrashingExecutorOperator
    }
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=executor_operators)
    with self.assertRaises(FakeError):
      _ = test_launcher.launch()

  def testLauncher_ExecutionFailedViaReturnCode(self):
    # In the case that the executor failed and raises an execption.
    # An Execution will be published.
    self.reloadPipelineWithNewRunId()
    LauncherTest.fakeUpstreamOutputs(self._mlmd_connection, self._example_gen,
                                     self._transform)
    executor_operators = {
        _PYTHON_CLASS_EXECUTABLE_SPEC: _FakeErrorExecutorOperator
    }
    test_launcher = launcher.Launcher(
        pipeline_node=self._trainer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_executor_operators=executor_operators)

    with self.assertRaisesRegex(
        Exception,
        'Execution .* failed with error code .* and error message .*'):
      _ = test_launcher.launch()

    with self._mlmd_connection as m:
      artifacts = m.store.get_artifacts_by_type('Model')
      self.assertEmpty(artifacts)
      executions = m.store.get_executions()
      self.assertProtoPartiallyEquals(
          """
          id: 3
          last_known_state: FAILED
          custom_properties {
            key: '__execution_result__'
            value {
              string_value: '{\\n  "resultMessage": "execution canceled.",\\n  "code": 1\\n}'
            }
          }
          """,
          executions[-1],
          ignored_fields=[
              'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_with_CustomDriver_NewSpan(self):
    self.reloadPipelineWithNewRunId()
    test_launcher = launcher.Launcher(
        pipeline_node=self._example_gen,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_driver_spec=self._custom_driver_spec,
        custom_executor_operators=self._test_executor_operators)
    _ = test_launcher.launch()

    with self._mlmd_connection as m:
      [artifact] = m.store.get_artifacts_by_type('Examples')
      self.assertProtoPartiallyEquals(
          """
          id: 1
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_example_gen:output_examples:0"
            }
          }
          custom_properties {
            key: "span"
            value {
              int_value: 2
            }
          }
          custom_properties {
            key: "version"
            value {
              int_value: 0
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifact,
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_with_CustomDriver_ExistingSpan(self):
    LauncherTest.fakeExampleGenOutput(self._mlmd_connection, self._example_gen,
                                      2, 1)
    self.reloadPipelineWithNewRunId()
    test_launcher = launcher.Launcher(
        pipeline_node=self._example_gen,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._trainer_executor_spec,
        custom_driver_spec=self._custom_driver_spec,
        custom_executor_operators=self._test_executor_operators)
    _ = test_launcher.launch()
    self.assertEqual(test_launcher._executor_operator._exec_properties, {
        'span': 2,
        'version': 2
    })

    with self._mlmd_connection as m:
      artifact = m.store.get_artifacts_by_type('Examples')[1]
      self.assertProtoPartiallyEquals(
          """
          id: 2
          custom_properties {
            key: "name"
            value {
              string_value: ":test_run_%d:my_example_gen:output_examples:0"
            }
          }
          custom_properties {
            key: "span"
            value {
              int_value: 2
            }
          }
          custom_properties {
            key: "version"
            value {
              int_value: 2
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""" % self._pipeline_run_id_counter,
          artifact,
          ignored_fields=[
              'type_id', 'uri', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

  def testLauncher_importer_node(self):
    mock_import_node_handler_class = mock.create_autospec(
        system_node_handler.SystemNodeHandler)
    mock_import_node_handler = mock.create_autospec(
        system_node_handler.SystemNodeHandler, instance=True)
    mock_import_node_handler_class.return_value = mock_import_node_handler
    expected_execution_info = data_types.ExecutionInfo()
    expected_execution_info.execution_id = 123
    mock_import_node_handler.run.return_value = expected_execution_info
    launcher._SYSTEM_NODE_HANDLERS[
        'tfx.dsl.components.common.importer.Importer'] = (
            mock_import_node_handler_class)
    test_launcher = launcher.Launcher(
        pipeline_node=self._importer,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec)
    execution_info = test_launcher.launch()
    mock_import_node_handler.run.assert_called_once_with(
        self._mlmd_connection, self._importer, self._pipeline_info,
        self._pipeline_runtime_spec)
    self.assertEqual(execution_info, expected_execution_info)

  def testLauncher_resolver_node(self):
    mock_resolver_node_handler_class = mock.create_autospec(
        system_node_handler.SystemNodeHandler)
    mock_resolver_node_handler = mock.create_autospec(
        system_node_handler.SystemNodeHandler, instance=True)
    mock_resolver_node_handler_class.return_value = mock_resolver_node_handler
    expected_execution_info = data_types.ExecutionInfo()
    expected_execution_info.execution_id = 123
    mock_resolver_node_handler.run.return_value = expected_execution_info
    launcher._SYSTEM_NODE_HANDLERS[
        'tfx.dsl.components.common.resolver.Resolver'] = (
            mock_resolver_node_handler_class)
    test_launcher = launcher.Launcher(
        pipeline_node=self._resolver,
        mlmd_connection=self._mlmd_connection,
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec)
    execution_info = test_launcher.launch()
    mock_resolver_node_handler.run.assert_called_once_with(
        self._mlmd_connection, self._resolver, self._pipeline_info,
        self._pipeline_runtime_spec)
    self.assertEqual(execution_info, expected_execution_info)


if __name__ == '__main__':
  tf.test.main()
