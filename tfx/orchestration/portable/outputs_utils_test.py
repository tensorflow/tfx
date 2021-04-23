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
"""Tests for tfx.orchestration.portable.output_utils."""
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration.portable import outputs_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types.value_artifact import ValueArtifact
from tfx.utils import test_case_utils

from google.protobuf import text_format

_PIPELINE_INFO = text_format.Parse("""
  id: "test_pipeline"
""", pipeline_pb2.PipelineInfo())

_PIPELINE_NODE = text_format.Parse(
    """
  node_info {
    id: "test_node"
  }
  outputs {
    outputs {
      key: "output_1"
      value {
        artifact_spec {
          type {
            id: 1
            name: "test_type_1"
            properties {
              key: "int_prop"
              value: INT
            }
            properties {
              key: "string_prop"
              value: STRING
            }
            properties {
              key: "float_prop"
              value: DOUBLE
            }
          }
          additional_properties {
            key: "int_prop"
            value {
              field_value {
                int_value: 42
              }
            }
          }
          additional_properties {
            key: "string_prop"
            value {
              field_value {
                string_value: "foo"
              }
            }
          }
          additional_properties {
            key: "float_prop"
            value {
              field_value {
                double_value: 0.5
              }
            }
          }
          additional_custom_properties {
            key: "float_custom_prop"
            value {
              field_value {
                double_value: 0.25
              }
            }
          }
          additional_custom_properties {
            key: "int_custom_prop"
            value {
              field_value {
                int_value: 21
              }
            }
          }
          additional_custom_properties {
            key: "string_custom_prop"
            value {
              field_value {
                string_value: "bar"
              }
            }
          }
        }
      }
    }
    outputs {
      key: "output_2"
      value {
        artifact_spec {
          type {
            id: 2
            name: "test_type_2"
          }
        }
      }
    }
   outputs {
      key: "output_3"
      value {
        artifact_spec {
          type {
            id: 3
            name: "String"
          }
        }
      }
    }
  }
""", pipeline_pb2.PipelineNode())


class OutputUtilsTest(test_case_utils.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    pipeline_runtime_spec = pipeline_pb2.PipelineRuntimeSpec()
    pipeline_runtime_spec.pipeline_root.field_value.string_value = self.tmp_dir
    pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        'test_run_0')
    self._pipeline_runtime_spec = pipeline_runtime_spec

  def _output_resolver(self, execution_mode=pipeline_pb2.Pipeline.SYNC):
    return outputs_utils.OutputsResolver(
        pipeline_node=_PIPELINE_NODE,
        pipeline_info=_PIPELINE_INFO,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        execution_mode=execution_mode)

  @parameterized.parameters(
      (pipeline_pb2.Pipeline.SYNC, 'test_pipeline:test_run_0:test_node'),
      (pipeline_pb2.Pipeline.ASYNC, 'test_pipeline:test_node'))
  def testGenerateOutputArtifacts(self, exec_mode, artifact_name_prefix):
    output_artifacts = self._output_resolver(
        exec_mode).generate_output_artifacts(1)
    self.assertIn('output_1', output_artifacts)
    self.assertIn('output_2', output_artifacts)
    self.assertIn('output_3', output_artifacts)
    self.assertLen(output_artifacts['output_1'], 1)
    self.assertLen(output_artifacts['output_2'], 1)
    self.assertLen(output_artifacts['output_3'], 1)

    artifact_1 = output_artifacts['output_1'][0]
    self.assertRegex(artifact_1.uri, '.*/test_node/output_1/1')
    self.assertRegex(artifact_1.name, artifact_name_prefix + ':output_1:0')
    self.assertProtoEquals(
        """
        id: 1
        name: "test_type_1"
        properties {
          key: "int_prop"
          value: INT
        }
        properties {
          key: "string_prop"
          value: STRING
        }
        properties {
          key: "float_prop"
          value: DOUBLE
        }
        """, artifact_1.artifact_type)
    self.assertLen(artifact_1.mlmd_artifact.properties, 3)
    # `name` is already a custom property of the artifact and 3 will be added.
    self.assertLen(artifact_1.mlmd_artifact.custom_properties, 4)
    self.assertEqual(artifact_1.int_prop, 42)
    self.assertEqual(artifact_1.float_prop, 0.5)
    self.assertEqual(artifact_1.string_prop, 'foo')
    self.assertEqual(artifact_1.get_int_custom_property('int_custom_prop'), 21)
    self.assertEqual(
        artifact_1.get_string_custom_property('string_custom_prop'), 'bar')
    self.assertEqual(
        artifact_1.get_float_custom_property('float_custom_prop'), 0.25)

    artifact_2 = output_artifacts['output_2'][0]
    self.assertRegex(artifact_2.uri, '.*/test_node/output_2/1')
    self.assertRegex(artifact_2.name, artifact_name_prefix + ':output_2:0')
    self.assertProtoEquals(
        """
        id: 2
        name: "test_type_2"
        """, artifact_2.artifact_type)
    self.assertEmpty(artifact_2.mlmd_artifact.properties)
    # `name` custom property is automatically generated.
    self.assertLen(artifact_2.mlmd_artifact.custom_properties, 1)

    artifact_3 = output_artifacts['output_3'][0]
    self.assertRegex(artifact_3.uri, '.*/test_node/output_3/1/value')
    self.assertRegex(artifact_3.name, artifact_name_prefix + ':output_3:0')
    self.assertProtoEquals("""
        id: 3
        name: "String"
        """, artifact_3.artifact_type)

  def testGetExecutorOutputUri(self):
    executor_output_uri = self._output_resolver().get_executor_output_uri(1)
    self.assertRegex(
        executor_output_uri,
        '.*/test_node/.system/executor_execution/1/executor_output.pb')
    # Verify that executor_output_uri is writable.
    with fileio.open(executor_output_uri, mode='wb') as f:
      executor_output = execution_result_pb2.ExecutorOutput()
      f.write(executor_output.SerializeToString())

  def testGetStatefulWorkingDir(self):
    stateful_working_dir = (
        self._output_resolver().get_stateful_working_directory())
    self.assertRegex(stateful_working_dir,
                     '.*/test_node/.system/stateful_working_dir/test_run_0')
    self.assertTrue(fileio.exists(stateful_working_dir))

  @parameterized.parameters(pipeline_pb2.Pipeline.SYNC,
                            pipeline_pb2.Pipeline.ASYNC)
  def testGetStatefulWorkingDirWithExecutionId(self, exec_mode):
    stateful_working_dir = (
        self._output_resolver(exec_mode).get_stateful_working_directory(1))
    self.assertRegex(stateful_working_dir,
                     '.*/test_node/.system/stateful_working_dir/1')
    fileio.exists(stateful_working_dir)

  def testGetStatefulWorkingDirAsyncRaisesWithoutExecutionId(self):
    with self.assertRaisesRegex(ValueError,
                                'Cannot create stateful working dir'):
      self._output_resolver(
          pipeline_pb2.Pipeline.ASYNC).get_stateful_working_directory()

  def testGetTmpDir(self):
    tmp_dir = self._output_resolver().make_tmp_dir(1)
    fileio.exists(tmp_dir)
    self.assertRegex(tmp_dir,
                     '.*/test_node/.system/executor_execution/1/.temp/')

  def testMakeOutputDirsAndRemoveOutputDirs(self):
    output_artifacts = self._output_resolver().generate_output_artifacts(1)
    outputs_utils.make_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        if isinstance(artifact, ValueArtifact):
          self.assertFalse(fileio.isdir(artifact.uri))
        else:
          self.assertTrue(fileio.isdir(artifact.uri))
        self.assertTrue(fileio.exists(artifact.uri))

    outputs_utils.remove_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        self.assertFalse(fileio.exists(artifact.uri))

  def testRemoveStatefulWorkingDirSucceeded(self):
    stateful_working_dir = (
        self._output_resolver().get_stateful_working_directory())
    self.assertTrue(fileio.exists(stateful_working_dir))

    outputs_utils.remove_stateful_working_dir(stateful_working_dir)
    self.assertFalse(fileio.exists(stateful_working_dir))

  def testRemoveStatefulWorkingDirNotFoundError(self):
    # removing a nonexisting path is an noop
    outputs_utils.remove_stateful_working_dir('/a/not/exist/path')

  @mock.patch.object(fileio, 'rmtree')
  def testRemoveStatefulWorkingDirOtherError(self, rmtree_fn):
    rmtree_fn.side_effect = ValueError('oops')
    with self.assertRaisesRegex(ValueError, 'oops'):
      outputs_utils.remove_stateful_working_dir('/a/fake/path')


if __name__ == '__main__':
  tf.test.main()
