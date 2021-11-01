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
"""Tests for tfx.orchestration.kubeflow.v2.parameter_utils."""

from concurrent import futures

import tensorflow as tf
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import parameter_utils


class ParameterUtilsTest(tf.test.TestCase):

  def _testAttachParametersInSingleThread(self, suffix: str):
    with parameter_utils.ParameterContext() as pc:
      parameter_utils.attach_parameter(
          data_types.RuntimeParameter(
              name='param1_in_{}'.format(suffix), ptype=int))
      parameter_utils.attach_parameter(
          data_types.RuntimeParameter(
              name='param2_in_{}'.format(suffix), ptype=int))
    self.assertLen(pc.parameters, 2)
    self.assertEqual(pc.parameters[0].name, 'param1_in_{}'.format(suffix))
    self.assertEqual(pc.parameters[1].name, 'param2_in_{}'.format(suffix))

  def testAttachParameters(self):
    with parameter_utils.ParameterContext() as pc:
      param1 = data_types.RuntimeParameter(name='test_param_1', ptype=int)
      parameter_utils.attach_parameter(param1)
      param2 = data_types.RuntimeParameter(name='test_param_2', ptype=str)
      parameter_utils.attach_parameter(param2)
      param3 = data_types.RuntimeParameter(name='test_param_3', ptype=float)
      parameter_utils.attach_parameter(param3)

    self.assertListEqual([param1, param2, param3], pc.parameters)

  def testAttachParametersInMultiThreads(self):
    with futures.ThreadPoolExecutor() as pool:
      future1 = pool.submit(
          self._testAttachParametersInSingleThread, suffix='thread-1')
      future2 = pool.submit(
          self._testAttachParametersInSingleThread, suffix='thread-2')
      future1.result()
      future2.result()

  def testFailWhenNotRunningUnderContext(self):
    param = data_types.RuntimeParameter(name='test_param', ptype=int)
    with self.assertRaisesRegex(
        RuntimeError,
        r'attach_parameter\(\) must run under ParameterContext\.'):
      parameter_utils.attach_parameter(param)


if __name__ == '__main__':
  tf.test.main()
