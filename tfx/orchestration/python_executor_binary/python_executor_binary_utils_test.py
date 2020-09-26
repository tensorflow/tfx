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
"""Tests for tfx.orchestration.python_executor_binary.python_executor_binary_utils."""

import tensorflow as tf
from tfx.orchestration.portable import base_executor_operator
from tfx.orchestration.python_executor_binary import python_executor_binary_utils
from tfx.types import artifact


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

    executor_output_uri = 'output/uri'
    stateful_working_dir = 'workding/dir'
    exec_properties = {
        'property1': 'value1',
        'property2': 'value2',
    }

    original = base_executor_operator.ExecutionInfo(
        input_dict={'input': [my_artifact]},
        output_dict={'output': [my_artifact]},
        exec_properties=exec_properties,
        executor_output_uri=executor_output_uri,
        stateful_working_dir=stateful_working_dir,
    )

    serialized = python_executor_binary_utils.serialize_execution_info(original)
    rehydrated = python_executor_binary_utils.deserialize_execution_info(
        serialized)

    self.CheckArtifactDict(rehydrated.input_dict, {'input': [my_artifact]})
    self.CheckArtifactDict(rehydrated.output_dict, {'output': [my_artifact]})
    self.assertEqual(rehydrated.exec_properties, exec_properties)
    self.assertEqual(rehydrated.executor_output_uri, executor_output_uri)
    self.assertEqual(rehydrated.stateful_working_dir, stateful_working_dir)


if __name__ == '__main__':
  tf.test.main()
