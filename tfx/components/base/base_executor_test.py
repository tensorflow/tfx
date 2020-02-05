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
"""Tests for tfx.components.base.base_executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text

from apache_beam.options.pipeline_options import DirectOptions
from apache_beam.options.pipeline_options import StandardOptions
import tensorflow.compat.v1 as tf

from tfx import types
from tfx.components.base import base_executor


class _TestExecutor(base_executor.BaseExecutor):
  """Fake executor for testing purpose only."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    pass


class BaseExecutorTest(tf.test.TestCase):

  def testBeamSettings(self):
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=['--runner=DirectRunner'])
    executor = _TestExecutor(executor_context)
    options = executor._make_beam_pipeline().options.view_as(StandardOptions)
    self.assertEqual('DirectRunner', options.runner)

    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=['--direct_num_workers=2'])
    executor = _TestExecutor(executor_context)
    options = executor._make_beam_pipeline().options.view_as(DirectOptions)
    self.assertEqual(2, options.direct_num_workers)


if __name__ == '__main__':
  tf.test.main()
