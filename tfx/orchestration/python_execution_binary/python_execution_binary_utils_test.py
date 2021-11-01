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
"""Tests for tfx.orchestration.python_execution_binary.python_execution_binary_utils."""
import tensorflow as tf
from tfx.orchestration.portable import data_types
from tfx.orchestration.python_execution_binary import python_execution_binary_utils
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyTypeName'
  PROPERTIES = {
      'int1': artifact.Property(type=artifact.PropertyType.INT),
  }


class PythonExecutorBinaryUtilsTest(tf.test.TestCase):

  def CheckArtifactDict(self, a, b):
    self.assertEqual(a.keys(), b.keys())
    for k, a_v in a.items():
      b_v = b[k]
      for a_artifact, b_artifact in zip(a_v, b_v):
        self.assertDictEqual(a_artifact.to_json_dict(),
                             b_artifact.to_json_dict())

  def testExecutionInfoSerialization(self):
    my_artifact = _MyArtifact()
    my_artifact.int1 = 111

    execution_output_uri = 'output/uri'
    stateful_working_dir = 'workding/dir'
    exec_properties = {
        'property1': 'value1',
        'property2': 'value2',
    }
    pipeline_info = pipeline_pb2.PipelineInfo(id='my_pipeline')
    pipeline_node = text_format.Parse(
        """
        node_info {
          id: 'my_node'
        }
        """, pipeline_pb2.PipelineNode())

    original = data_types.ExecutionInfo(
        input_dict={'input': [my_artifact]},
        output_dict={'output': [my_artifact]},
        exec_properties=exec_properties,
        execution_output_uri=execution_output_uri,
        stateful_working_dir=stateful_working_dir,
        pipeline_info=pipeline_info,
        pipeline_node=pipeline_node)

    serialized = python_execution_binary_utils.serialize_execution_info(
        original)
    rehydrated = python_execution_binary_utils.deserialize_execution_info(
        serialized)

    self.CheckArtifactDict(rehydrated.input_dict, {'input': [my_artifact]})
    self.CheckArtifactDict(rehydrated.output_dict, {'output': [my_artifact]})
    self.assertEqual(rehydrated.exec_properties, exec_properties)
    self.assertEqual(rehydrated.execution_output_uri, execution_output_uri)
    self.assertEqual(rehydrated.stateful_working_dir, stateful_working_dir)
    self.assertProtoEquals(rehydrated.pipeline_info, original.pipeline_info)
    self.assertProtoEquals(rehydrated.pipeline_node, original.pipeline_node)

  def testExecutableSpecSerialization(self):
    python_executable_spec = text_format.Parse(
        """
        class_path: 'path_to_my_class'
        extra_flags: '--flag=my_flag'
        """, executable_spec_pb2.PythonClassExecutableSpec())
    python_serialized = python_execution_binary_utils.serialize_executable_spec(
        python_executable_spec)
    python_rehydrated = python_execution_binary_utils.deserialize_executable_spec(
        python_serialized)
    self.assertProtoEquals(python_rehydrated, python_executable_spec)

    beam_executable_spec = text_format.Parse(
        """
        python_executor_spec {
          class_path: 'path_to_my_class'
          extra_flags: '--flag1=1'
        }
        beam_pipeline_args: '--arg=my_beam_pipeline_arg'
        """, executable_spec_pb2.BeamExecutableSpec())
    beam_serialized = python_execution_binary_utils.serialize_executable_spec(
        beam_executable_spec)
    beam_rehydrated = python_execution_binary_utils.deserialize_executable_spec(
        beam_serialized, with_beam=True)
    self.assertProtoEquals(beam_rehydrated, beam_executable_spec)

  def testMlmdConnectionConfigSerialization(self):
    connection_config = text_format.Parse(
        """
        sqlite {
          filename_uri: 'my_file_uri'
        }
        """, metadata_store_pb2.ConnectionConfig())

    rehydrated_connection_config = python_execution_binary_utils.deserialize_mlmd_connection_config(
        python_execution_binary_utils.serialize_mlmd_connection_config(
            connection_config))

    self.assertProtoEquals(rehydrated_connection_config, connection_config)


if __name__ == '__main__':
  tf.test.main()
