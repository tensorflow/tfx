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
"""Tests for tfx.tools.cli.builder.buildspec."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import yaml

from tfx.tools.cli.container_builder import buildspec


class BuildSpecTest(tf.test.TestCase):

  def _generate_temp_path(self):
    return os.path.join(self.create_tempdir().full_path, 'generated')

  def _read_expected_buildspec(self, name):
    default_buildspec_path = os.path.join(
        os.path.dirname(__file__), 'testdata', name)
    with open(default_buildspec_path, 'r') as f:
      golden_buildspec = yaml.safe_load(f)
    return golden_buildspec

  def _read_generated_buildspec(self, build_spec):
    with open(build_spec.filename, 'r') as f:
      return yaml.safe_load(f)

  def test_generate_clean(self):
    spec = buildspec.BuildSpec.load_default(
        filename=self._generate_temp_path(),
        target_image='gcr.io/test:dev')

    self.assertEqual(self._read_generated_buildspec(spec),
                     self._read_expected_buildspec('test_buildspec'))

  def test_generate_custom(self):
    spec = buildspec.BuildSpec.load_default(
        filename=self._generate_temp_path(),
        build_context='/path/to/somewhere',
        target_image='gcr.io/test:dev',
        dockerfile_name='dev.Dockerfile')

    self.assertEqual(self._read_generated_buildspec(spec),
                     self._read_expected_buildspec('test_buildspec_custom'))

if __name__ == '__main__':
  tf.test.main()
