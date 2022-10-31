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
import os
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration.portable import outputs_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_property
from tfx.types import standard_artifacts
from tfx.types.value_artifact import ValueArtifact
from tfx.utils import test_case_utils

from google.protobuf import text_format

# TODO(b/241861488): Remove safeguard once fully supported by MLMD.
artifact_property.ENABLE_PROTO_PROPERTIES = True

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
            properties {
              key: "proto_prop"
              value: PROTO
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
          additional_properties {
            key: "proto_prop"
            value {
              field_value {
                proto_value {
                  type_url: "type.googleapis.com/google.protobuf.Value"
                  value: "\\032\\003aaa"
                }
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
          additional_custom_properties {
            key: "proto_custom_prop"
            value {
              field_value {
                proto_value {
                  type_url: "type.googleapis.com/google.protobuf.Value"
                  value: "\\032\\003bbb"
                }
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
    outputs {
      key: "output_4"
      value {
        artifact_spec {
          type {
            id: 4
            name: "Integer_Metrics"
          }
        }
      }
    }
    outputs {
      key: "output_5"
      value {
        artifact_spec {
          type {
            id: 5
            name: "External_Artifact"
          }
          external_artifact_uris: "/external_directory_1/123"
          external_artifact_uris: "/external_directory_2/456"
        }
      }
    }
    outputs {
      key: "output_6"
      value {
        artifact_spec {
          type {
            id: 6
            name: "String"
          }
          external_artifact_uris: "/external_directory_3/789"
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

  def _get_external_uri_for_test(self, uri):
    return os.path.join(self.tmp_dir, os.path.relpath(uri, '/'))

  @parameterized.parameters(
      (pipeline_pb2.Pipeline.SYNC, 'test_pipeline:test_run_0:test_node:1'),
      (pipeline_pb2.Pipeline.ASYNC, 'test_pipeline:test_node:1'))
  def testGenerateOutputArtifacts(self, exec_mode, artifact_name_prefix):
    output_artifacts = self._output_resolver(
        exec_mode).generate_output_artifacts(1)
    self.assertIn('output_1', output_artifacts)
    self.assertIn('output_2', output_artifacts)
    self.assertIn('output_3', output_artifacts)
    self.assertIn('output_4', output_artifacts)
    self.assertIn('output_5', output_artifacts)
    self.assertIn('output_6', output_artifacts)
    self.assertLen(output_artifacts['output_1'], 1)
    self.assertLen(output_artifacts['output_2'], 1)
    self.assertLen(output_artifacts['output_3'], 1)
    self.assertLen(output_artifacts['output_4'], 1)
    # If there are multiple external_artifact_uris,
    # it has to make multiple artifacts of the same number.
    self.assertLen(output_artifacts['output_5'], 2)
    self.assertLen(output_artifacts['output_6'], 1)

    artifact_1 = output_artifacts['output_1'][0]
    self.assertRegex(artifact_1.uri, '.*/test_node/output_1/1')
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
        properties {
          key: "proto_prop"
          value: PROTO
        }
        """, artifact_1.artifact_type)
    self.assertLen(artifact_1.mlmd_artifact.properties, 4)
    self.assertLen(artifact_1.mlmd_artifact.custom_properties, 4)
    self.assertEqual(artifact_1.int_prop, 42)
    self.assertEqual(artifact_1.float_prop, 0.5)
    self.assertEqual(artifact_1.string_prop, 'foo')
    self.assertEqual(artifact_1.proto_prop.string_value, 'aaa')
    self.assertEqual(artifact_1.get_int_custom_property('int_custom_prop'), 21)
    self.assertEqual(
        artifact_1.get_string_custom_property('string_custom_prop'), 'bar')
    self.assertEqual(
        artifact_1.get_float_custom_property('float_custom_prop'), 0.25)
    self.assertEqual(
        artifact_1.get_proto_custom_property('proto_custom_prop').string_value,
        'bbb')

    artifact_2 = output_artifacts['output_2'][0]
    self.assertRegex(artifact_2.uri, '.*/test_node/output_2/1')
    self.assertProtoEquals(
        """
        id: 2
        name: "test_type_2"
        """, artifact_2.artifact_type)
    self.assertEmpty(artifact_2.mlmd_artifact.properties)
    self.assertEmpty(artifact_2.mlmd_artifact.custom_properties)

    artifact_3 = output_artifacts['output_3'][0]
    self.assertRegex(artifact_3.uri, '.*/test_node/output_3/1/value')
    self.assertProtoEquals("""
        id: 3
        name: "String"
        """, artifact_3.artifact_type)

    artifact_4 = output_artifacts['output_4'][0]
    self.assertRegex(artifact_4.uri, '.*/test_node/output_4/1/value')
    self.assertProtoEquals("""
        id: 4
        name: "Integer_Metrics"
        """, artifact_4.artifact_type)

    artifact_5_0 = output_artifacts['output_5'][0]
    self.assertEqual(artifact_5_0.uri, '/external_directory_1/123')
    self.assertProtoEquals("""
        id: 5
        name: "External_Artifact"
        """, artifact_5_0.artifact_type)

    artifact_5_1 = output_artifacts['output_5'][1]
    self.assertEqual(artifact_5_1.uri, '/external_directory_2/456')
    self.assertProtoEquals("""
        id: 5
        name: "External_Artifact"
        """, artifact_5_1.artifact_type)

    artifact_6 = output_artifacts['output_6'][0]
    self.assertEqual(artifact_6.uri, '/external_directory_3/789')
    self.assertProtoEquals("""
        id: 6
        name: "String"
        """, artifact_6.artifact_type)

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

  def testMakeClearAndRemoveOutputDirs(self):
    output_artifacts = self._output_resolver().generate_output_artifacts(1)
    outputs_utils.make_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        if artifact.is_external:
          continue
        if isinstance(artifact, ValueArtifact):
          self.assertFalse(fileio.isdir(artifact.uri))
        else:
          self.assertTrue(fileio.isdir(artifact.uri))
          with fileio.open(os.path.join(artifact.uri, 'output'), 'w') as f:
            f.write('')
        self.assertTrue(fileio.exists(artifact.uri))

    outputs_utils.clear_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        if artifact.is_external:
          continue
        if not isinstance(artifact, ValueArtifact):
          self.assertEqual(fileio.listdir(artifact.uri), [])

    outputs_utils.remove_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        if artifact.is_external:
          continue
        self.assertFalse(fileio.exists(artifact.uri))

  def testMakeOutputDirsArtifactAlreadyExists(self):
    output_artifacts = self._output_resolver().generate_output_artifacts(1)
    outputs_utils.make_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        if artifact.is_external:
          continue
        if isinstance(artifact, ValueArtifact):
          with fileio.open(artifact.uri, 'w') as f:
            f.write('test')
        else:
          with fileio.open(os.path.join(artifact.uri, 'output'), 'w') as f:
            f.write('test')
    outputs_utils.make_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        if artifact.is_external:
          continue
        if isinstance(artifact, ValueArtifact):
          with fileio.open(artifact.uri, 'r') as f:
            self.assertEqual(f.read(), 'test')
        else:
          with fileio.open(os.path.join(artifact.uri, 'output'), 'r') as f:
            self.assertEqual(f.read(), 'test')

  def testOmitLifeCycleManagementForExternalArtifact(self):
    """Test that it omits lifecycle management for external artifacts."""
    external_artifacts = self._output_resolver().generate_output_artifacts(1)
    for key, artifact_list in external_artifacts.items():
      external_artifacts[key] = [
          artifact for artifact in artifact_list if artifact.is_external
      ]
      for artifact in external_artifacts[key]:
        artifact.uri = self._get_external_uri_for_test(artifact.uri)

    outputs_utils.make_output_dirs(external_artifacts)
    for _, artifact_list in external_artifacts.items():
      for artifact in artifact_list:
        # make_output_dirs method doesn't affect the external uris.
        self.assertFalse(fileio.exists(artifact.uri))

        # Make new directory and file for next test.
        if isinstance(artifact, ValueArtifact):
          artifact_dir = os.path.dirname(artifact.uri)
          fileio.makedirs(artifact_dir)
          with fileio.open(artifact.uri, 'w') as f:
            f.write('test')
        else:
          fileio.makedirs(artifact.uri)
          with fileio.open(os.path.join(artifact.uri, 'output'),
                           'w') as f:
            f.write('test')

    outputs_utils.clear_output_dirs(external_artifacts)
    for _, artifact_list in external_artifacts.items():
      for artifact in artifact_list:
        # clear_output_dirs method doesn't affect the external uris.
        if isinstance(artifact, ValueArtifact):
          with fileio.open(artifact.uri, 'r') as f:
            self.assertEqual(f.read(), 'test')
        else:
          with fileio.open(os.path.join(artifact.uri, 'output'),
                           'r') as f:
            self.assertEqual(f.read(), 'test')

    outputs_utils.remove_output_dirs(external_artifacts)
    for _, artifact_list in external_artifacts.items():
      for artifact in artifact_list:
        # remove_output_dirs method doesn't affect the external uris.
        self.assertTrue(fileio.exists(artifact.uri))

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

  def testPopulateOutputArtifact(self):
    executor_output = execution_result_pb2.ExecutorOutput()
    output_dict = {'output_key': [standard_artifacts.Model()]}
    outputs_utils.populate_output_artifact(executor_output, output_dict)
    self.assertProtoEquals(
        """
        output_artifacts {
          key: "output_key"
          value {
            artifacts {
            }
          }
        }
        """, executor_output)

  def testPopulateExecProperties(self):
    executor_output = execution_result_pb2.ExecutorOutput()
    exec_properties = {'string_value': 'string', 'int_value': 1}
    outputs_utils.populate_exec_properties(executor_output, exec_properties)
    self.assertProtoEquals(
        """
        execution_properties {
          key: "string_value"
          value {
            string_value: "string"
          }
        }
        execution_properties {
          key: "int_value"
          value {
            int_value: 1
          }
        }
        """, executor_output)

if __name__ == '__main__':
  tf.test.main()
