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

import copy
import tensorflow as tf
from tfx.orchestration import publisher
from tfx.utils import types


class PublisherTest(tf.test.TestCase):

  def setUp(self):
    self._mock_metadata = tf.test.mock.Mock()
    self._mock_metadata.publish_execution = tf.test.mock.Mock()
    self._input_dict = {
        'input_data': [types.TfxArtifact(type_name='InputType')],
    }
    self._output_dict = {
        'output_data': [types.TfxArtifact(type_name='OutputType')],
    }
    self._execution_id = 100

  def test_prepare_execution_completed(self):
    input_dict = copy.deepcopy(self._input_dict)
    output_dict = copy.deepcopy(self._output_dict)

    p = publisher.Publisher(metadata_handler=self._mock_metadata)
    p.publish_execution(
        self._execution_id, input_dict, output_dict, use_cached_results=False)
    self._mock_metadata.publish_execution.assert_called_with(
        execution_id=self._execution_id,
        input_dict=input_dict,
        output_dict=output_dict,
        state='completed')

  def test_prepare_execution_cached(self):
    input_dict = copy.deepcopy(self._input_dict)
    output_dict = copy.deepcopy(self._output_dict)

    p = publisher.Publisher(metadata_handler=self._mock_metadata)
    p.publish_execution(
        self._execution_id, input_dict, output_dict, use_cached_results=True)
    self._mock_metadata.publish_execution.assert_called_with(
        execution_id=self._execution_id,
        input_dict=input_dict,
        output_dict=output_dict,
        state='skipped')


if __name__ == '__main__':
  tf.test.main()
