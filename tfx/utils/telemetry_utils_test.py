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
"""Tests for tfx.utils.telemetry_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
# Standard Imports
import tensorflow as tf

from tfx import version
from tfx.utils import telemetry_utils


class TelemetryUtilsTest(tf.test.TestCase):

  def testMakeBeamLabelsArgs(self):
    """Test for make_beam_labels_args."""
    beam_pipeline_args = telemetry_utils.make_beam_labels_args()
    self.assertListEqual([
        '--labels',
        'tfx_py_version=%d-%d' %
        (sys.version_info.major, sys.version_info.minor),
        '--labels',
        'tfx_version=%s' % version.__version__.replace('.', '-'),
    ], beam_pipeline_args)

  def testScopedLabels(self):
    """Test for scoped_labels."""
    orig_labels = telemetry_utils.get_labels_dict()
    with telemetry_utils.scoped_labels({'foo': 'bar'}):
      self.assertDictEqual(telemetry_utils.get_labels_dict(),
                           dict({'foo': 'bar'}, **orig_labels))
      with telemetry_utils.scoped_labels({'inner': 'baz'}):
        self.assertDictEqual(
            telemetry_utils.get_labels_dict(),
            dict({
                'foo': 'bar',
                'inner': 'baz'
            }, **orig_labels))


if __name__ == '__main__':
  tf.test.main()
