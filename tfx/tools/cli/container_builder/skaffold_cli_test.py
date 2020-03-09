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
"""Tests for tfx.tools.cli.container_builder.skaffold_cli."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import mock
import tensorflow as tf

from tfx.tools.cli.container_builder import skaffold_cli

SAMPLE_SKAFFOLD_OUTPUT = b'{"builds":[{"imageName":"gcr.io/my-test/test","tag":"gcr.io/my-test/test:latest@sha256:f5ee0ecb19eb5dc970f15290f2c47c10d23e303d381aebef91929c2df2ce5004"}]}'


class SkaffoldCliTest(tf.test.TestCase):

  @mock.patch.object(subprocess, 'run')
  @mock.patch.object(os.path, 'exists')
  @mock.patch.object(
      subprocess, 'check_output', return_value=SAMPLE_SKAFFOLD_OUTPUT)
  def testSkaffoldBuild(self, mock_run, mock_exists, mock_check_output):
    cli = skaffold_cli.SkaffoldCli()
    sha256 = cli.build()
    self.assertEqual(
        sha256,
        'sha256:f5ee0ecb19eb5dc970f15290f2c47c10d23e303d381aebef91929c2df2ce5004'
    )


if __name__ == '__main__':
  tf.test.main()
