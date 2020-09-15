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

import mock
import tensorflow as tf
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

_executed_components = []
_component_executors = {}
_component_drivers = {}


# TODO(b/162980675): When PythonExecutorOperator is implemented. We don't
# Need to Fake the whole FakeComponentAsDoFn. Instead, just fake or mock
# executors.
class _FakeComponentAsDoFn(beam_dag_runner._PipelineNodeAsDoFn):

  def __init__(self,
               pipeline_node: pipeline_pb2.PipelineNode,
               mlmd_connection: metadata.Metadata,
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
               executor_spec: Optional[message.Message],
               custom_driver_spec: Optional[message.Message]):
    self._component_id = pipeline_node.node_info.id
    _component_executors[self._component_id] = executor_spec
    _component_drivers[self._component_id] = custom_driver_spec

  def _run_component(self):
    _executed_components.append(self._component_id)


class BeamDagRunnerTest(test_utils.TfxTest):

  def setUp(self):
    super(BeamDagRunnerTest, self).setUp()
    # Setup pipelines
    self._pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata',
            'pipeline_for_launcher_test.pbtxt'), self._pipeline)
    _executed_components.clear()
    _component_executors.clear()
    _component_drivers.clear()

  @mock.patch.multiple(
      beam_dag_runner,
      _PipelineNodeAsDoFn=_FakeComponentAsDoFn,
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
                text_format.Parse(
                    'class_path: "tfx.components.trainer_executor"',
                    _PythonClassExecutableSpec())
        })
    self.assertEqual(
        _component_drivers, {
            'my_example_gen':
                text_format.Parse(
                    'class_path: "tfx.components.example_gen_driver"',
                    _PythonClassExecutableSpec()),
            'my_transform': None,
            'my_trainer': None
        })
    self.assertEqual(_executed_components,
                     ['my_example_gen', 'my_transform', 'my_trainer'])


if __name__ == '__main__':
  tf.test.main()
