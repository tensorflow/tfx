# Lint as: python2, python3
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
"""Tests for tfx.tools.cli.handler.template_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tfx.tools.cli import labels
from tfx.tools.cli.handler import template_handler


class TemplateHandlerTest(tf.test.TestCase):

  def testList(self):
    templates = template_handler.list_template()
    self.assertNotEqual(templates, [])
    self.assertIn('taxi', templates)

  def testCopy(self):
    test_dir = self.create_tempdir().full_path
    flags = {
        labels.MODEL: 'taxi',
        labels.DESTINATION_PATH: test_dir,
        labels.PIPELINE_NAME: 'my_pipeline'
    }
    template_handler.copy_template(flags)
    copied_files = os.listdir(test_dir)
    self.assertNotEqual(copied_files, [])
    self.assertContainsSubset(['__init__.py'], copied_files)

  def testEscapePipelineName(self):
    # pylint: disable=protected-access
    self.assertEqual('x', template_handler._sanitize_pipeline_name('x'))
    self.assertEqual('\\\\x\\\'\\"',
                     template_handler._sanitize_pipeline_name('\\x\'"'))
    self.assertEqual('a\\/b', template_handler._sanitize_pipeline_name('a/b'))
    # pylint: enable=protected-access

if __name__ == '__main__':
  tf.test.main()
