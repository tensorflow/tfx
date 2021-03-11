# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.publisher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx import types
from tfx import version
from tfx.orchestration import data_types
from tfx.orchestration import publisher


class _InputType(types.Artifact):
  TYPE_NAME = 'InputType'


class _OutputType(types.Artifact):
  TYPE_NAME = 'OutputType'


class PublisherTest(tf.test.TestCase):

  def setUp(self):
    super(PublisherTest, self).setUp()
    self._mock_metadata = tf.compat.v1.test.mock.Mock()
    self._mock_metadata.publish_execution = tf.compat.v1.test.mock.Mock()
    self._output_dict = {
        'output_data': [_OutputType()],
    }
    self._exec_properties = {'k': 'v'}
    self._pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline', pipeline_root='/tmp', run_id='my_run_id')
    self._component_info = data_types.ComponentInfo(
        component_type='a.b.c',
        component_id='my_component',
        pipeline_info=self._pipeline_info)

  def testPrepareExecutionComplete(self):
    p = publisher.Publisher(metadata_handler=self._mock_metadata)
    p.publish_execution(
        component_info=self._component_info,
        output_artifacts=self._output_dict,
        exec_properties=self._exec_properties)
    self._mock_metadata.publish_execution.assert_called_with(
        component_info=self._component_info,
        output_artifacts=self._output_dict,
        exec_properties=self._exec_properties)
    self.assertEqual(
        self._output_dict['output_data'][0].get_string_custom_property(
            'tfx_version'), version.__version__)


if __name__ == '__main__':
  tf.test.main()
