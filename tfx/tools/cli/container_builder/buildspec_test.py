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
from tfx.tools.cli.container_builder import buildspec
from tfx.tools.cli.container_builder import labels
import yaml


class BuildSpecTest(tf.test.TestCase):

  def setUp(self):
    super(BuildSpecTest, self).setUp()
    self._default_buildspec_path = os.path.join(
        os.path.dirname(__file__), 'testdata', 'test_buildspec')

  def test_generate_clean(self):
    output_path = os.path.join(self.create_tempdir().full_path, 'generated')
    build_spec = buildspec.BuildSpec.load_default(
        filename=output_path,
        target_image='gcr.io/test:dev',
        dockerfile_name=labels.DOCKERFILE_NAME)
    with open(self._default_buildspec_path, 'r') as f:
      golden_buildspec = yaml.safe_load(f)
    with open(build_spec.filename, 'r') as f:
      generated_buildspec = yaml.safe_load(f)

    self.assertEqual(generated_buildspec, golden_buildspec)

  def test_read(self):
    build_spec = buildspec.BuildSpec(self._default_buildspec_path)
    self.assertEqual(build_spec.filename, self._default_buildspec_path)
    self.assertEqual(build_spec.build_context, '.')
    self.assertEqual(build_spec.target_image, 'gcr.io/test')
    self.assertEqual(build_spec.target_image_tag, 'dev')

if __name__ == '__main__':
  tf.test.main()
