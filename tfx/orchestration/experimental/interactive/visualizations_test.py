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
"""Tests for tfx.orchestration.experimental.interactive.visualizations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import mock

from six.moves import builtins
import tensorflow as tf

from tfx.orchestration.experimental.interactive import visualizations
from tfx.types import standard_artifacts


class VisualizationsTest(tf.test.TestCase):

  def setUp(self):
    super(VisualizationsTest, self).setUp()
    builtins.__dict__['__IPYTHON__'] = True

  def tearDown(self):
    del builtins.__dict__['__IPYTHON__']
    super(VisualizationsTest, self).tearDown()

  @mock.patch('tfx.orchestration.experimental.interactive.'
              'visualizations.get_registry')
  def testVisualizationRegistrationAndUsage(self, *unused_mocks):
    registry = visualizations.ArtifactVisualizationRegistry()
    visualizations.get_registry = mock.MagicMock(return_value=registry)
    mock_object = mock.MagicMock()

    class MyVisualization(visualizations.ArtifactVisualization):

      # Arbitrary artifact type class.
      ARTIFACT_TYPE = standard_artifacts.Examples

      def display(self, unused_artifact):
        mock_object('foo')

    visualizations.get_registry().register(MyVisualization)
    self.assertIs(
        MyVisualization,
        visualizations.get_registry().get_visualization(
            standard_artifacts.Examples.TYPE_NAME).__class__)


if __name__ == '__main__':
  tf.test.main()
