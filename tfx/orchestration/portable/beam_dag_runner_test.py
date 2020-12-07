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

import tensorflow as tf
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.portable import beam_dag_runner
from tfx.orchestration.portable import test_utils
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import message
from google.protobuf import text_format

_PythonClassExecutableSpec = executable_spec_pb2.PythonClassExecutableSpec
_LOCAL_DEPLOYMENT_CONFIG = text_format.Parse("""
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
        python_class_executable_spec {
          class_path: "tfx.components.trainer_executor"
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
""", local_deployment_config_pb2.LocalDeploymentConfig())

_INTERMEDIATE_DEPLOYMENT_CONFIG = text_format.Parse("""
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
        [type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec] {
          class_path: "tfx.components.trainer_executor"
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
""", pipeline_pb2.IntermediateDeploymentConfig())

_executed_components = []
_component_executors = {}
_component_drivers = {}
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
    pipeline_run = None
    for context in pipeline_node.contexts.contexts:
      if context.type.name == constants.PIPELINE_RUN_ID_PARAMETER_NAME:
        pipeline_run = context.name.field_value.string_value
    _conponent_to_pipeline_run[self._node_id] = pipeline_run

  def _run_component(self):
    _executed_components.append(self._node_id)


class BeamDagRunnerTest(test_utils.TfxTest):

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
    _conponent_to_pipeline_run.clear()

  def testRunWithLocalDeploymentConfig(self):
    self._pipeline.deployment_config.Pack(_INTERMEDIATE_DEPLOYMENT_CONFIG)
    beam_dag_runner.BeamDagRunner._PIPELINE_NODE_DO_FN_CLS = _FakeComponentAsDoFn
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
                text_format.Parse(
                    'class_path: "tfx.components.trainer_executor"',
                    _PythonClassExecutableSpec()),
            'my_importer': None,
        })
    self.assertEqual(
        _component_drivers, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_driver"',
                    _PythonClassExecutableSpec()),
            'my_transform': None,
            'my_trainer': None,
            'my_importer': None,
        })
    # 'my_importer' has no upstream and can be executed in any order.
    self.assertIn('my_importer', _executed_components)
    _executed_components.remove('my_importer')
    self.assertEqual(_executed_components,
                     ['my_example_gen', 'my_transform', 'my_trainer'])
    # Verifies that every component gets a not-None pipeline_run.
    self.assertTrue(all(_conponent_to_pipeline_run.values()))

  def testRunWithIntermediateDeploymentConfig(self):
    self._pipeline.deployment_config.Pack(_LOCAL_DEPLOYMENT_CONFIG)
    beam_dag_runner.BeamDagRunner._PIPELINE_NODE_DO_FN_CLS = _FakeComponentAsDoFn
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
                text_format.Parse(
                    'class_path: "tfx.components.trainer_executor"',
                    _PythonClassExecutableSpec()),
            'my_importer': None,
        })
    self.assertEqual(
        _component_drivers, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_driver"',
                    _PythonClassExecutableSpec()),
            'my_transform': None,
            'my_trainer': None,
            'my_importer': None,
        })
    # 'my_importer' has no upstream and can be executed in any order.
    self.assertIn('my_importer', _executed_components)
    _executed_components.remove('my_importer')
    self.assertEqual(_executed_components,
                     ['my_example_gen', 'my_transform', 'my_trainer'])
    # Verifies that every component gets a not-None pipeline_run.
    self.assertTrue(all(_conponent_to_pipeline_run.values()))

if __name__ == '__main__':
  tf.test.main()
