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
"""Tests for tfx.orchestration.portable.beam_dag_runner."""
import os
from typing import Optional

from unittest import mock
import tensorflow as tf
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.beam.legacy import beam_dag_runner as legacy_beam_dag_runner
from tfx.orchestration.config import pipeline_config
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.proto.orchestration import platform_config_pb2
from tfx.utils import test_case_utils

from google.protobuf import message
from google.protobuf import text_format

_PythonClassExecutableSpec = executable_spec_pb2.PythonClassExecutableSpec
_ContainerExecutableSpec = executable_spec_pb2.ContainerExecutableSpec
_DockerPlatformConfig = platform_config_pb2.DockerPlatformConfig

_LOCAL_DEPLOYMENT_CONFIG = text_format.Parse(
    """
    executor_specs {
      key: "my_example_gen"
      value {
        python_class_executable_spec {
          class_path: "tfx.components.example_gen_executor"
        }
      }
    }
    executor_specs {
      key: "my_transform"
      value {
        python_class_executable_spec {
          class_path: "tfx.components.transform_executor"
        }
      }
    }
    executor_specs {
      key: "my_trainer"
      value {
        container_executable_spec {
          image: "path/to/docker/image"
        }
      }
    }
    custom_driver_specs {
      key: "my_example_gen"
      value {
        python_class_executable_spec {
          class_path: "tfx.components.example_gen_driver"
        }
      }
    }
    metadata_connection_config {
      fake_database {}
    }
    node_level_platform_configs {
      key: "my_trainer"
      value {
        docker_platform_config {
          docker_server_url: "docker/server/url"
        }
      }
    }
""", local_deployment_config_pb2.LocalDeploymentConfig())

_INTERMEDIATE_DEPLOYMENT_CONFIG = text_format.Parse(
    """
    executor_specs {
      key: "my_example_gen"
      value {
        [type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec] {
          class_path: "tfx.components.example_gen_executor"
        }
      }
    }
    executor_specs {
      key: "my_transform"
      value {
        [type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec] {
          class_path: "tfx.components.transform_executor"
        }
      }
    }
    executor_specs {
      key: "my_trainer"
      value {
        [type.googleapis.com/tfx.orchestration.executable_spec.ContainerExecutableSpec] {
          image: "path/to/docker/image"
        }
      }
    }
    custom_driver_specs {
      key: "my_example_gen"
      value {
        [type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec] {
          class_path: "tfx.components.example_gen_driver"
        }
      }
    }
    metadata_connection_config {
      [type.googleapis.com/ml_metadata.ConnectionConfig] {
        fake_database {}
      }
    }
    node_level_platform_configs {
      key: "my_trainer"
      value {
        [type.googleapis.com/tfx.orchestration.platform_config.DockerPlatformConfig] {
          docker_server_url: "docker/server/url"
        }
      }
    }
""", pipeline_pb2.IntermediateDeploymentConfig())

_executed_components = []
_component_executors = {}
_component_drivers = {}
_component_platform_configs = {}
_conponent_to_pipeline_run = {}


# TODO(b/162980675): When PythonExecutorOperator is implemented. We don't
# Need to Fake the whole FakeComponentAsDoFn. Instead, just fake or mock
# executors.
class _FakeComponentAsDoFn(beam_dag_runner.PipelineNodeAsDoFn):

  def __init__(self, pipeline_node: pipeline_pb2.PipelineNode,
               mlmd_connection_config: metadata.ConnectionConfigType,
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
               executor_spec: Optional[message.Message],
               custom_driver_spec: Optional[message.Message],
               deployment_config: Optional[message.Message]):
    super().__init__(pipeline_node, mlmd_connection_config, pipeline_info,
                     pipeline_runtime_spec, executor_spec, custom_driver_spec,
                     deployment_config)
    _component_executors[self._node_id] = executor_spec
    _component_drivers[self._node_id] = custom_driver_spec
    _component_platform_configs[self._node_id] = self._extract_platform_config(
        self._deployment_config, self._node_id)
    pipeline_run = None
    for context in pipeline_node.contexts.contexts:
      if context.type.name == constants.PIPELINE_RUN_CONTEXT_TYPE_NAME:
        pipeline_run = context.name.field_value.string_value
    _conponent_to_pipeline_run[self._node_id] = pipeline_run

  def _run_node(self):
    _executed_components.append(self._node_id)


class BeamDagRunnerTest(test_case_utils.TfxTest):

  def setUp(self):
    super(BeamDagRunnerTest, self).setUp()
    # Setup pipelines
    self._pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata',
            'pipeline_for_beam_dag_runner_test.pbtxt'), self._pipeline)
    _executed_components.clear()
    _component_executors.clear()
    _component_drivers.clear()
    _component_platform_configs.clear()
    _conponent_to_pipeline_run.clear()

  @mock.patch.multiple(
      beam_dag_runner.BeamDagRunner,
      _PIPELINE_NODE_DO_FN_CLS=_FakeComponentAsDoFn,
  )
  def testRunWithLocalDeploymentConfig(self):
    self._pipeline.deployment_config.Pack(_LOCAL_DEPLOYMENT_CONFIG)
    beam_dag_runner.BeamDagRunner().run(self._pipeline)
    self.assertEqual(
        _component_executors, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_executor"',
                    _PythonClassExecutableSpec()),
            'my_transform':
                text_format.Parse(
                    'class_path: "tfx.components.transform_executor"',
                    _PythonClassExecutableSpec()),
            'my_trainer':
                text_format.Parse('image: "path/to/docker/image"',
                                  _ContainerExecutableSpec()),
            'my_importer':
                None,
        })
    self.assertEqual(
        _component_drivers, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_driver"',
                    _PythonClassExecutableSpec()),
            'my_transform':
                None,
            'my_trainer':
                None,
            'my_importer':
                None,
        })
    self.assertEqual(
        _component_platform_configs, {
            'my_example_gen':
                None,
            'my_transform':
                None,
            'my_trainer':
                text_format.Parse('docker_server_url: "docker/server/url"',
                                  _DockerPlatformConfig()),
            'my_importer':
                None,
        })
    # 'my_importer' has no upstream and can be executed in any order.
    self.assertIn('my_importer', _executed_components)
    _executed_components.remove('my_importer')
    self.assertEqual(_executed_components,
                     ['my_example_gen', 'my_transform', 'my_trainer'])
    # Verifies that every component gets a not-None pipeline_run.
    self.assertTrue(all(_conponent_to_pipeline_run.values()))

  @mock.patch.multiple(
      beam_dag_runner.BeamDagRunner,
      _PIPELINE_NODE_DO_FN_CLS=_FakeComponentAsDoFn,
  )
  def testRunWithIntermediateDeploymentConfig(self):
    self._pipeline.deployment_config.Pack(_INTERMEDIATE_DEPLOYMENT_CONFIG)
    beam_dag_runner.BeamDagRunner().run(self._pipeline)
    self.assertEqual(
        _component_executors, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_executor"',
                    _PythonClassExecutableSpec()),
            'my_transform':
                text_format.Parse(
                    'class_path: "tfx.components.transform_executor"',
                    _PythonClassExecutableSpec()),
            'my_trainer':
                text_format.Parse('image: "path/to/docker/image"',
                                  _ContainerExecutableSpec()),
            'my_importer':
                None,
        })
    self.assertEqual(
        _component_drivers, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_driver"',
                    _PythonClassExecutableSpec()),
            'my_transform':
                None,
            'my_trainer':
                None,
            'my_importer':
                None,
        })
    self.assertEqual(
        _component_platform_configs, {
            'my_example_gen':
                None,
            'my_transform':
                None,
            'my_trainer':
                text_format.Parse('docker_server_url: "docker/server/url"',
                                  _DockerPlatformConfig()),
            'my_importer':
                None,
        })
    # 'my_importer' has no upstream and can be executed in any order.
    self.assertIn('my_importer', _executed_components)
    _executed_components.remove('my_importer')
    self.assertEqual(_executed_components,
                     ['my_example_gen', 'my_transform', 'my_trainer'])
    # Verifies that every component gets a not-None pipeline_run.
    self.assertTrue(all(_conponent_to_pipeline_run.values()))

  def testLegacyBeamDagRunnerConstruction(self):
    self.assertIsInstance(beam_dag_runner.BeamDagRunner(),
                          beam_dag_runner.BeamDagRunner)

    # Test that the legacy Beam DAG runner is used when a PipelineConfig is
    # specified.
    config = pipeline_config.PipelineConfig()
    runner = beam_dag_runner.BeamDagRunner(config=config)
    self.assertIs(runner.__class__, legacy_beam_dag_runner.BeamDagRunner)
    self.assertIs(runner._config, config)

    # Test that the legacy Beam DAG runner is used when beam_orchestrator_args
    # is specified.
    beam_orchestrator_args = ['--my-beam-option']
    runner = beam_dag_runner.BeamDagRunner(
        beam_orchestrator_args=beam_orchestrator_args)
    self.assertIs(runner.__class__, legacy_beam_dag_runner.BeamDagRunner)
    self.assertIs(runner._beam_orchestrator_args, beam_orchestrator_args)

    # Test that the legacy Beam DAG runner is used when both a PipelineConfig
    # and beam_orchestrator_args are specified.
    config = pipeline_config.PipelineConfig()
    beam_orchestrator_args = ['--my-beam-option']
    runner = beam_dag_runner.BeamDagRunner(
        config=config, beam_orchestrator_args=beam_orchestrator_args)
    self.assertIs(runner.__class__, legacy_beam_dag_runner.BeamDagRunner)
    self.assertIs(runner._config, config)
    self.assertIs(runner._beam_orchestrator_args, beam_orchestrator_args)


if __name__ == '__main__':
  tf.test.main()
