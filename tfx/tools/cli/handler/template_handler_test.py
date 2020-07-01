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
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler import template_handler


class TemplateHandlerTest(tf.test.TestCase):

  _PLACEHOLDER_TEST_DATA_BEFORE = """
  from %s import mmm
  pipeline_name = '{{PIPELINE_NAME}}'
  """ % ('tfx.experimental.templates.taxi.parent',)

  _PLACEHOLDER_TEST_DATA_AFTER = """
  from parent import mmm
  pipeline_name = 'dummy'
  """

  def testList(self):
    templates = template_handler.list_template()
    self.assertNotEqual(templates, [])
    self.assertIn('taxi', templates)

  def testCopy(self):
    test_dir = self.create_tempdir().full_path
    pipeline_name = 'my_pipeline'
    flags = {
        labels.MODEL: 'taxi',
        labels.DESTINATION_PATH: test_dir,
        labels.PIPELINE_NAME: pipeline_name
    }
    template_handler.copy_template(flags)
    copied_files = os.listdir(test_dir)
    self.assertNotEqual(copied_files, [])
    self.assertContainsSubset(['__init__.py', 'beam_dag_runner.py'],
                              copied_files)
    self.assertFalse(os.path.exists(os.path.join(test_dir, 'e2e_tests')))
    self.assertTrue(os.path.exists(os.path.join(test_dir, 'data', 'data.csv')))

    with open(os.path.join(test_dir, 'pipeline', 'configs.py')) as fp:
      configs_py_content = fp.read()
    self.assertIn(pipeline_name, configs_py_content)

  def testEscapePipelineName(self):
    # pylint: disable=protected-access
    self.assertEqual('x', template_handler._sanitize_pipeline_name('x'))
    self.assertEqual('\\\\x\\\'\\"',
                     template_handler._sanitize_pipeline_name('\\x\'"'))
    self.assertEqual('a\\/b', template_handler._sanitize_pipeline_name('a/b'))
    # pylint: enable=protected-access

  def testReplacePlaceHolder(self):
    pipeline_name = 'dummy'
    src = self.create_tempfile()
    dst = self.create_tempfile()
    # pylint: disable=protected-access
    replace_dict = {
        template_handler._IMPORT_FROM_PACKAGE:
            template_handler._IMPORT_FROM_LOCAL_DIR,
        template_handler._PLACEHOLDER_PIPELINE_NAME:
            pipeline_name,
    }
    src.write_text(self._PLACEHOLDER_TEST_DATA_BEFORE)
    template_handler._copy_and_replace_placeholder_file(src.full_path,
                                                        dst.full_path,
                                                        replace_dict)
    # pylint: enable=protected-access
    self.assertEqual(dst.read_text(), self._PLACEHOLDER_TEST_DATA_AFTER)


if __name__ == '__main__':
  tf.test.main()
