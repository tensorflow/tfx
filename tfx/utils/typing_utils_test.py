# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.typing_utils."""
from typing import Any

import tensorflow as tf
import tfx.types
from tfx.types import standard_artifacts
from tfx.utils import typing_utils


class TypingUtilsTest(tf.test.TestCase):

  def _model(self):
    return standard_artifacts.Model()

  def test_artifact_multimap_pylint(self):

    def get_artifact(input_dict: typing_utils.ArtifactMultiMap):
      return input_dict['model'][0]

    def get_type_name(artifact: tfx.types.Artifact):
      return artifact.type_name

    # No pytype complain
    input_dict = {'model': [self._model()]}
    self.assertEqual(get_type_name(get_artifact(input_dict)), 'Model')

  def test_is_artifact_multimap(self):

    def yes(value: Any):
      self.assertTrue(typing_utils.is_artifact_multimap(value))

    def no(value: Any):
      self.assertFalse(typing_utils.is_artifact_multimap(value))

    yes({})
    yes({'model': []})
    yes({'model': [self._model()]})
    yes({'model': [self._model(), self._model()]})
    no({'model': [self._model(), 'not an artifact']})
    no({'model': self._model()})
    no({123: [self._model()]})

  def test_is_list_of_artifact_multimap(self):

    def yes(value: Any):
      self.assertTrue(typing_utils.is_list_of_artifact_multimap(value))

    def no(value: Any):
      self.assertFalse(typing_utils.is_list_of_artifact_multimap(value))

    yes([])
    yes([{}])
    yes([{}, {}])
    yes([{'model': []}])
    yes([{'model': [self._model()]}])
    yes([{'model': [self._model(), self._model()]}])
    no([self._model()])
    no([{'model': self._model()}])


if __name__ == '__main__':
  tf.test.main()
