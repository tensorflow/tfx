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
"""Tests for tfx.dsl.components.base.executor_spec."""

import tensorflow as tf
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec


class _TestSpecWithoutEncode(executor_spec.ExecutorSpec):
  pass

  def copy(self):
    return self


class _DummyExecutor(base_executor.BaseExecutor):
  pass


class ExecutorSpecTest(tf.test.TestCase):

  def testNotImplementedError(self):
    with self.assertRaisesRegex(
        NotImplementedError,
        '_TestSpecWithoutEncode does not support encoding into IR.'):
      _TestSpecWithoutEncode().encode()

  def testExecutorClassSpecCopy(self):
    spec = executor_spec.ExecutorClassSpec(_DummyExecutor)
    spec.add_extra_flags('a')
    spec_copy = spec.copy()
    del spec
    self.assertProtoEquals(
        """
        class_path: "__main__._DummyExecutor"
        extra_flags: "a"
        """,
        spec_copy.encode())

  def testBeamExecutorSpecCopy(self):
    spec = executor_spec.BeamExecutorSpec(_DummyExecutor)
    spec.add_extra_flags('a')
    spec.add_beam_pipeline_args('b')
    spec_copy = spec.copy()
    del spec
    self.assertProtoEquals(
        """
        python_executor_spec: {
            class_path: "__main__._DummyExecutor"
            extra_flags: "a"
        }
        beam_pipeline_args: "b"
        beam_pipeline_args_placeholders {
          value {
            string_value: "b"
          }
        }
        """, spec_copy.encode())

  def testExecutorContainerSpecCopy(self):
    spec = executor_spec.ExecutorContainerSpec(
        image='path/to:image', command=['command'], args=['args'])
    spec_copy = spec.copy()
    del spec
    self.assertEqual(spec_copy.image, 'path/to:image')
    self.assertEqual(spec_copy.command, ['command'])
    self.assertEqual(spec_copy.args, ['args'])

if __name__ == '__main__':
  tf.test.main()
