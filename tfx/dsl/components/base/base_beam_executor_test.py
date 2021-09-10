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
"""Tests for tfx.dsl.components.base.base_beam_executor."""

import sys
from typing import Any, Dict, List
from unittest import mock

from apache_beam.options.pipeline_options import DirectOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import StandardOptions
import tensorflow as tf

from tfx import types
from tfx import version
from tfx.components.statistics_gen.executor import Executor as StatisticsGenExecutor
from tfx.dsl.components.base import base_beam_executor


class _TestExecutor(base_beam_executor.BaseBeamExecutor):
  """Fake executor for testing purpose only."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    pass


class BaseBeamExecutorTest(tf.test.TestCase):

  def testBeamSettings(self):
    executor_context = base_beam_executor.BaseBeamExecutor.Context(
        beam_pipeline_args=['--runner=DirectRunner'])
    executor = _TestExecutor(executor_context)
    options = executor._make_beam_pipeline().options.view_as(StandardOptions)
    self.assertEqual('DirectRunner', options.view_as(StandardOptions).runner)
    # Verify labels.
    self.assertListEqual([
        'tfx_executor=third_party_executor',
        'tfx_py_version=%d-%d' %
        (sys.version_info.major, sys.version_info.minor),
        'tfx_version=%s' % version.__version__.replace('.', '-'),
    ],
                         options.view_as(GoogleCloudOptions).labels)

    executor_context = base_beam_executor.BaseBeamExecutor.Context(
        beam_pipeline_args=['--direct_num_workers=2'])
    executor = StatisticsGenExecutor(executor_context)
    options = executor._make_beam_pipeline().options.view_as(DirectOptions)
    self.assertEqual(2, options.direct_num_workers)
    # Verify labels.
    self.assertListEqual([
        'tfx_executor=tfx-components-statistics_gen-executor-executor',
        'tfx_py_version=%d-%d' %
        (sys.version_info.major, sys.version_info.minor),
        'tfx_version=%s' % version.__version__.replace('.', '-'),
    ],
                         options.view_as(GoogleCloudOptions).labels)

  def testCustomBeamMakePipelineFn(self):
    mock_fn = mock.MagicMock()
    executor_context = base_beam_executor.BaseBeamExecutor.Context(
        make_beam_pipeline_fn=mock_fn)
    executor = _TestExecutor(executor_context)
    executor._make_beam_pipeline()
    mock_fn.assert_called_once_with()

if __name__ == '__main__':
  tf.test.main()
