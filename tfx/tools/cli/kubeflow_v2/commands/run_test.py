# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for Kubeflow V2 run commands."""
# TODO(b/169094706): Add kokoro test coverage for this test.

import codecs
import locale
import os
import sys
from unittest import mock

from click import testing as click_testing
import tensorflow as tf

# Fake GCP project ID, API key, docker image tag and job name under test.
# _TEST_API_KEY = 'fake-api-key'
# _TEST_PROJECT_ID = 'fake-gcp-project'
# _TEST_IMAGE = 'gcr.io/fake-image:fake-tag'
# _TEST_JOB_NAME = 'taxi-pipeline-1'


# TODO(b/169094706): re-surrect the tests when the unified client becomes
# available.
class RunTest(tf.test.TestCase):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    super().setUp()
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    sys.modules['handler_factory'] = mock.Mock()


if __name__ == '__main__':
  tf.test.main()
