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
"""Tests for tfx.utils.types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard Imports

import mock
import tensorflow as tf
from tensorflow.python.platform import tf_logging  # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import types


class TypesTest(tf.test.TestCase):

  def testTfxtypeDeprecated(self):
    print(deprecation._PRINTED_WARNING)
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      types.TfxType('FakeType')
      warn_mock.assert_called_once()
      self.assertIn(
          'tfx.utils.types.TfxType has been renamed to tfx.types.Artifact',
          warn_mock.call_args[0][5])
    # Reset deprecation message sentinel.
    print(deprecation._PRINTED_WARNING)
    del deprecation._PRINTED_WARNING[artifact.Artifact]

  def testTfxartifactDeprecated(self):
    print(deprecation._PRINTED_WARNING)
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      types.TfxArtifact('FakeType')
      warn_mock.assert_called_once()
      self.assertIn(
          'tfx.utils.types.TfxArtifact has been renamed to tfx.types.Artifact',
          warn_mock.call_args[0][5])
    # Reset deprecation message sentinel.
    print(deprecation._PRINTED_WARNING)
    del deprecation._PRINTED_WARNING[artifact.Artifact]

  def testParseTfxTypeDictDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      self.assertEqual({}, types.parse_tfx_type_dict('{}'))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.parse_tfx_type_dict has been renamed to',
                    warn_mock.call_args[0][5])

  def testJsonifyTfxTypeDictDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      self.assertEqual('{}', types.jsonify_tfx_type_dict({}))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.jsonify_tfx_type_dict has been renamed to',
                    warn_mock.call_args[0][5])

  def testGetSingleInstanceDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      my_artifact = artifact.Artifact('TestType')
      self.assertIs(my_artifact, types.get_single_instance([my_artifact]))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.get_single_instance has been renamed to',
                    warn_mock.call_args[0][5])

  def testGetSingleUriDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      my_artifact = artifact.Artifact('TestType')
      my_artifact.uri = '123'
      self.assertEqual('123', types.get_single_uri([my_artifact]))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.get_single_uri has been renamed to',
                    warn_mock.call_args[0][5])

  def testGetSplitUriDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      my_artifact = standard_artifacts.Examples()
      my_artifact.uri = '123'
      my_artifact.split_names = artifact_utils.encode_split_names(['train'])
      self.assertEqual('123/train', types.get_split_uri([my_artifact], 'train'))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.get_split_uri has been renamed to',
                    warn_mock.call_args[0][5])


if __name__ == '__main__':
  tf.test.main()
