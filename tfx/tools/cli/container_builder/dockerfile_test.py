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
"""Tests for tfx.tools.cli.builder.dockerfile."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import filecmp
import os

import tensorflow as tf

from tfx import version
from tfx.tools.cli.container_builder import dockerfile
from tfx.tools.cli.container_builder import labels
from tfx.utils import test_case_utils


_FAKE_VERSION = '0.23.0'

_test_dockerfile_content = '''FROM tensorflow/tfx:%s
WORKDIR /pipeline
COPY ./ ./
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"''' % _FAKE_VERSION


class DockerfileTest(test_case_utils.TfxTest):

  def setUp(self):
    super(DockerfileTest, self).setUp()
    self._testdata_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'testdata')
    # change to a temporary working dir such that
    # there is no setup.py in the working dir.
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

    self._test_dockerfile = os.path.abspath('.test_dockerfile')
    with open(self._test_dockerfile, 'w') as f:
      f.writelines(_test_dockerfile_content)

  @contextlib.contextmanager
  def _patchVersion(self, ver):
    old_version = version.__version__
    old_base_image = labels.BASE_IMAGE
    version.__version__ = ver
    labels.BASE_IMAGE = 'tensorflow/tfx:%s' % ver
    yield
    labels.BASE_IMAGE = old_base_image
    version.__version__ = old_version

  def testGenerate(self):
    generated_dockerfile_path = labels.DOCKERFILE_NAME
    with self._patchVersion(_FAKE_VERSION):
      dockerfile.Dockerfile(filename=generated_dockerfile_path)
      self.assertTrue(
          filecmp.cmp(self._test_dockerfile, generated_dockerfile_path))

  def testGenerateWithBaseOverride(self):
    generated_dockerfile_path = labels.DOCKERFILE_NAME
    dockerfile.Dockerfile(
        filename=generated_dockerfile_path,
        base_image='my_customized_image:latest')
    self.assertTrue(
        filecmp.cmp(
            os.path.join(self._testdata_dir, 'test_dockerfile_with_base'),
            generated_dockerfile_path))

  def testDevVersionRequirement(self):
    with self._patchVersion('0.23.0.dev'):
      with self.assertRaisesRegex(ValueError,
                                  'Cannot find a base image automatically'):
        dockerfile.Dockerfile(filename=labels.DOCKERFILE_NAME)


if __name__ == '__main__':
  tf.test.main()
