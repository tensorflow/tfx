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
import tensorflow as tf
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable import test_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format

_PIPELING_INFO = text_format.Parse("""
  id: "test_pipeline"
""", pipeline_pb2.PipelineInfo())

_PIPLINE_NODE = text_format.Parse(
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
  }
""", pipeline_pb2.PipelineNode())


class OutputUtilsTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()
    pipeline_runtime_spec = pipeline_pb2.PipelineRuntimeSpec()
    pipeline_runtime_spec.pipeline_root.field_value.string_value = self.tmp_dir
    pipeline_runtime_spec.pipeline_run_id.field_value.string_value = (
        'test_run_0')

    self._output_resolver = outputs_utils.OutputsResolver(
        pipeline_node=_PIPLINE_NODE,
        pipeline_info=_PIPELING_INFO,
        pipeline_runtime_spec=pipeline_runtime_spec)

  def testGenerateOutputArtifacts(self):
    output_artifacts = self._output_resolver.generate_output_artifacts(1)
    self.assertIn('output_1', output_artifacts)
    self.assertIn('output_2', output_artifacts)
    self.assertLen(output_artifacts['output_1'], 1)
    self.assertLen(output_artifacts['output_2'], 1)

    artifact_1 = output_artifacts['output_1'][0]
    self.assertRegex(artifact_1.uri, '.*/test_node/execution_1/output_1')
    self.assertRegex(artifact_1.name,
                     'test_pipeline:test_run_0:test_node:output_1:0')
    self.assertProtoEquals(
        """
        id: 1
        name: "test_type_1"
        """, artifact_1.artifact_type)

    artifact_2 = output_artifacts['output_2'][0]
    self.assertRegex(artifact_2.uri, '.*/test_node/execution_1/output_2')
    self.assertRegex(artifact_2.name,
                     'test_pipeline:test_run_0:test_node:output_2:0')
    self.assertProtoEquals(
        """
        id: 2
        name: "test_type_2"
        """, artifact_2.artifact_type)

  def testGetExecutorOutputUri(self):
    executor_output_uri = self._output_resolver.get_executor_output_uri(1)
    self.assertRegex(executor_output_uri,
                     '.*/test_node/execution_1/executor_output.pb')
    # Verify that executor_output_uri is writable.
    with tf.io.gfile.GFile(executor_output_uri, mode='w') as f:
      executor_output = execution_result_pb2.ExecutorOutput()
      f.write(executor_output.SerializeToString())

  def testGetWorkingDirectory(self):
    stateful_working_dir = (
        self._output_resolver.get_stateful_working_directory())
    self.assertRegex(stateful_working_dir,
                     '.*/test_node/test_run_0/stateful_working_dir')
    tf.io.gfile.exists(stateful_working_dir)
    # Mock the case of retry, verify that the same stateful_working_dir is
    # returned.
    stateful_working_dir = (
        self._output_resolver.get_stateful_working_directory())
    self.assertRegex(stateful_working_dir,
                     '.*/test_node/test_run_0/stateful_working_dir')

  def testMakeOutputDirsAndRemoveOutputDirs(self):
    output_artifacts = self._output_resolver.generate_output_artifacts(1)
    outputs_utils.make_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        self.assertTrue(tf.io.gfile.exists(artifact.uri))

    outputs_utils.remove_output_dirs(output_artifacts)
    for _, artifact_list in output_artifacts.items():
      for artifact in artifact_list:
        self.assertFalse(tf.io.gfile.exists(artifact.uri))


if __name__ == '__main__':
  tf.test.main()
