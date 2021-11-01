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

import codecs
import locale
import os
import re

from typing import List, Iterable, Tuple

from tfx.tools.cli.e2e import test_utils as cli_test_utils
from tfx.utils import io_utils
from tfx.utils import test_case_utils


class BaseEndToEndTest(test_case_utils.TfxTest):
  """Base class for end-to-end testing of TFX templates."""

  def setUp(self):
    super().setUp()

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

  def _runCli(self, args: List[str]) -> str:
    return cli_test_utils.run_cli(args)

  def _addAllComponents(self) -> str:
    """Change 'pipeline.py' file to put all components into the pipeline."""
    return self._uncomment(
        os.path.join('pipeline', 'pipeline.py'), ['components.append('])

  def _uncomment(self, filepath: str, expressions: Iterable[str]) -> str:
    """Update given file by uncommenting the `expression`."""
    replacements = [('# ' + s, s) for s in expressions]
    return self._replaceFileContent(filepath, replacements)

  def _comment(self, filepath: str, expressions: Iterable[str]) -> str:
    """Update given file by commenting out the `expression`."""
    replacements = [(s, '# ' + s) for s in expressions]
    return self._replaceFileContent(filepath, replacements)

  def _replaceFileContent(self, filepath: str,
                          replacements: Iterable[Tuple[str, str]]) -> str:
    """Update given file using `replacements`."""
    path = os.path.join(self._project_dir, filepath)
    with open(path) as fp:
      content = fp.read()
    for old, new in replacements:
      content = content.replace(old, new)
    io_utils.write_string_file(path, content)
    return path

  def _uncommentMultiLineVariables(self, filepath: str,
                                   variables: Iterable[str]) -> str:
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
          if re.match(r'# [\]\}\) ]', line):
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
    self.assertIn('Copying {} pipeline template'.format(model), result)


class BaseLocalEndToEndTest(BaseEndToEndTest):
  """Common tests for local engine."""

  def _getAllUnitTests(self):
    for root, _, files in os.walk(self._project_dir):
      base_dir = os.path.relpath(root, self._project_dir)
      if base_dir == '.':  # project_dir == root
        base_module = ''
      else:
        base_module = base_dir.replace(os.path.sep, '.') + '.'

      for filename in files:
        if filename.endswith('_test.py'):
          yield base_module + filename[:-3]

  def _create_pipeline(self):
    result = self._runCli([
        'pipeline',
        'create',
        '--engine',
        'local',
        '--pipeline_path',
        'local_runner.py',
    ])
    self.assertIn(
        'Pipeline "{}" created successfully.'.format(self._pipeline_name),
        result)

  def _update_pipeline(self):
    result = self._runCli([
        'pipeline',
        'update',
        '--engine',
        'local',
        '--pipeline_path',
        'local_runner.py',
    ])
    self.assertIn(
        'Pipeline "{}" updated successfully.'.format(self._pipeline_name),
        result)

  def _run_pipeline(self):
    self._runCli([
        'run',
        'create',
        '--engine',
        'local',
        '--pipeline_name',
        self._pipeline_name,
    ])

  def _copy_schema(self):
    self._runCli([
        'pipeline',
        'schema',
        '--engine',
        'local',
        '--pipeline_name',
        self._pipeline_name,
    ])
