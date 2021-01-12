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
"""E2E test utilities for templates."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os

from typing import Text, List, Iterable, Tuple

from absl import logging
from click import testing as click_testing

from tfx.tools.cli.cli_main import cli_group
from tfx.utils import io_utils
from tfx.utils import test_case_utils


class BaseEndToEndTest(test_case_utils.TfxTest):
  """Base class for end-to-end testing of TFX templates."""

  def setUp(self):
    super(BaseEndToEndTest, self).setUp()

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    # TODO(b/150100590) Delete this block after Python >=3.7
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    self._pipeline_name = 'TEMPLATE_E2E_TEST'
    self._project_dir = self.tmp_dir
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))
    self._temp_dir = os.path.join(self._project_dir, 'tmp')
    os.makedirs(self._temp_dir)

    # Initialize CLI runner.
    self._cli_runner = click_testing.CliRunner()

  def _runCli(self, args: List[Text]) -> click_testing.Result:
    logging.info('Running cli: %s', args)
    result = self._cli_runner.invoke(cli_group, args)
    logging.info('%s', result.output)
    if result.exit_code != 0:
      logging.error('Exit code from cli: %d, exception:%s', result.exit_code,
                    result.exception)
      logging.error('Traceback: %s', result.exc_info)

    return result

  def _addAllComponents(self) -> Text:
    """Change 'pipeline.py' file to put all components into the pipeline."""
    return self._uncomment(
        os.path.join('pipeline', 'pipeline.py'), ['components.append('])

  def _uncomment(self, filepath: Text, expressions: Iterable[Text]) -> Text:
    """Update given file by uncommenting the `expression`."""
    replacements = [('# ' + s, s) for s in expressions]
    return self._replaceFileContent(filepath, replacements)

  def _comment(self, filepath: Text, expressions: Iterable[Text]) -> Text:
    """Update given file by commenting out the `expression`."""
    replacements = [(s, '# ' + s) for s in expressions]
    return self._replaceFileContent(filepath, replacements)

  def _replaceFileContent(self, filepath: Text,
                          replacements: Iterable[Tuple[Text, Text]]) -> Text:
    """Update given file using `replacements`."""
    path = os.path.join(self._project_dir, filepath)
    with open(path) as fp:
      content = fp.read()
    for old, new in replacements:
      content = content.replace(old, new)
    io_utils.write_string_file(path, content)
    return path

  def _uncommentMultiLineVariables(self, filepath: Text,
                                   variables: Iterable[Text]) -> Text:
    """Update given file by uncommenting a variable.

    The variable should be defined in following form.
    # ....
    # VARIABLE_NAME = ...
    #   long indented line
    #
    #   long indented line
    # OTHER STUFF

    Above comments will become

    # ....
    VARIABLE_NAME = ...
      long indented line

      long indented line
    # OTHER STUFF

    Args:
      filepath: file to modify.
      variables: List of variables.

    Returns:
      Absolute path of the modified file.
    """
    path = os.path.join(self._project_dir, filepath)
    result = []
    commented_variables = ['# ' + variable + ' =' for variable in variables]
    in_variable_definition = False

    with open(path) as fp:
      for line in fp:
        if in_variable_definition:
          if line.startswith('#  ') or line.startswith('# }'):
            result.append(line[2:])
            continue
          elif line == '#\n':
            result.append(line[1:])
            continue
          else:
            in_variable_definition = False
        for commented_var in commented_variables:
          if line.startswith(commented_var):
            in_variable_definition = True
            result.append(line[2:])
            break
        else:
          # doesn't include a variable definition to uncomment.
          result.append(line)

    io_utils.write_string_file(path, ''.join(result))
    return path

  def _copyTemplate(self, model):
    result = self._runCli([
        'template',
        'copy',
        '--pipeline_name',
        self._pipeline_name,
        '--destination_path',
        self._project_dir,
        '--model',
        model,
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Copying {} pipeline template'.format(model), result.output)
