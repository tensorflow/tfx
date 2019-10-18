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

import filecmp
import os
import tempfile
import tensorflow as tf

from tfx.tools.cli.container_builder import dockerfile
from tfx.tools.cli.container_builder import labels


class DockerfileTest(tf.test.TestCase):

  def test_generate(self):
    # change to a temporary working dir such that there is no setup.py
    # in the working dir.
    old_working_dir = os.getcwd()
    tmp_working_dir = tempfile.mkdtemp()
    os.chdir(tmp_working_dir)

    test_dockerfile_name = 'test_dockerfile'
    default_dockerfile_path = os.path.join(
        os.path.dirname(__file__), 'testdata',
        test_dockerfile_name)
    generated_dockerfile_path = labels.DOCKERFILE_NAME
    dockerfile.Dockerfile(filename=generated_dockerfile_path)
    self.assertTrue(
        filecmp.cmp(default_dockerfile_path, generated_dockerfile_path))

    # clean up
    os.chdir(old_working_dir)

if __name__ == '__main__':
  tf.test.main()
