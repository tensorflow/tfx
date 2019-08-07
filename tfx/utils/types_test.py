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
from tfx.utils import types


class TypesTest(tf.test.TestCase):

  def test_tfxtype_deprecated(self):
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

  def test_tfxartifact_deprecated(self):
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

  def test_parse_tfx_type_dict_deprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      self.assertEqual({}, types.parse_tfx_type_dict('{}'))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.parse_tfx_type_dict has been renamed to',
                    warn_mock.call_args[0][5])

  def test_jsonify_tfx_type_dict_deprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      self.assertEqual('{}', types.jsonify_tfx_type_dict({}))
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.types.jsonify_tfx_type_dict has been renamed to',
                    warn_mock.call_args[0][5])


if __name__ == '__main__':
  tf.test.main()
