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
"""Base handler class."""

import abc
from importlib import machinery
from importlib import util as import_util
import json
import os
import subprocess
import sys
from typing import Any, Collection, Dict, List, Optional

import click

from tfx.dsl.components.base import base_driver
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import dag_runner_patcher
from tfx.utils import io_utils


class BaseHandler(abc.ABC):
  """Base Handler for CLI.

  Attributes:
    flags_dict: A dictionary with flags provided in a command.
  """

  def __init__(self, flags_dict: Dict[str, Any]):
    self.flags_dict = flags_dict
    self._handler_home_dir = self._get_handler_home()

  @abc.abstractmethod
  def create_pipeline(self) -> None:
    """Creates pipeline for the handler."""
    pass

  @abc.abstractmethod
  def update_pipeline(self) -> None:
    """Updates pipeline for the handler."""
    pass

  @abc.abstractmethod
  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    pass

  @abc.abstractmethod
  def delete_pipeline(self) -> None:
    """Deletes pipeline for the handler."""
    pass

  @abc.abstractmethod
  def compile_pipeline(self) -> None:
    """Compiles pipeline for the handler."""
    pass

  @abc.abstractmethod
  def create_run(self) -> None:
    """Runs a pipeline for the handler."""
    pass

  @abc.abstractmethod
  def delete_run(self) -> None:
    """Deletes a run."""
    pass

  @abc.abstractmethod
  def terminate_run(self) -> None:
    """Stops a run."""
    pass

  @abc.abstractmethod
  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    pass

  @abc.abstractmethod
  def get_run(self) -> None:
    """Checks run status."""
    pass

  def _check_pipeline_dsl_path(self) -> None:
    """Check if pipeline dsl path exists."""
    pipeline_dsl_path = self.flags_dict[labels.PIPELINE_DSL_PATH]
    if not fileio.exists(pipeline_dsl_path):
      sys.exit('Invalid pipeline path: {}'.format(pipeline_dsl_path))

  def execute_dsl(
      self, patcher: dag_runner_patcher.DagRunnerPatcher) -> Dict[str, Any]:
    """Execute DSL file with given patcher applied to the DSL Runner."""
    self._check_pipeline_dsl_path()
    dsl_path = self.flags_dict[labels.PIPELINE_DSL_PATH]

    with patcher.patch() as context:
      # Simluate python script execution.
      # - Need to add the script directory as a first entry of sys.path.
      # - Load the script as if we are in __main__ module.
      dir_path = os.path.dirname(os.path.realpath(dsl_path))
      sys.path.insert(0, dir_path)
      loader = machinery.SourceFileLoader('__main__', dsl_path)
      try:
        loader.exec_module(
            import_util.module_from_spec(
                import_util.spec_from_loader(loader.name, loader)))
      except SystemExit as system_exit:  # Swallow normal exit in absl.app.run()
        if system_exit.code != 0 and system_exit.code is not None:
          raise

      sys.path.pop(0)

      if not patcher.run_called:
        sys.exit('Cannot find ' + patcher.get_runner_class().__name__ +
                 '.run() in ' + dsl_path)
      return context

  def _get_handler_home(self) -> str:
    """Sets handler home.

    Returns:
      Path to handler home directory.
    """
    engine_flag = self.flags_dict[labels.ENGINE_FLAG]
    handler_home_dir = engine_flag.upper() + '_HOME'
    if handler_home_dir in os.environ:
      return os.environ[handler_home_dir]
    return os.path.join(os.environ['HOME'], 'tfx', engine_flag, '')

  def _get_deprecated_handler_home(self) -> str:
    """Sets old handler home for compatibility.

    Returns:
      Path to handler home directory.
    """
    engine_flag = self.flags_dict[labels.ENGINE_FLAG]
    handler_home_dir = engine_flag.upper() + '_HOME'
    if handler_home_dir in os.environ:
      return os.environ[handler_home_dir]
    return os.path.join(os.environ['HOME'], engine_flag, '')

  def _subprocess_call(self,
                       command: List[str],
                       env: Optional[Dict[str, Any]] = None) -> None:
    return_code = subprocess.call(command, env=env)
    if return_code != 0:
      sys.exit('Error while running "{}" '.format(' '.join(command)))

  def _format_table(self, header: Collection[Any],
                    data: Collection[Collection[Any]]):
    """Pretty-print the table (like tabluate library does)."""

    def _format_as_strings(items):
      return [f' {item} ' for item in items]

    header = _format_as_strings(header)
    data = [_format_as_strings(row) for row in data]

    max_widths = [len(s) for s in header]
    for row in data:
      max_widths = [
          max(c_max, len(item)) for c_max, item in zip(max_widths, row)
      ]

    def _make_line(record, widths, sep='|', fill=' '):
      return (sep + sep.join(
          item.ljust(width, fill) for item, width in zip(record, widths)) +
              sep + '\n')

    empty_data = [''] * len(header)  # empty data for horizontal line
    double_separator = _make_line(empty_data, max_widths, '+', '=')
    single_separator = _make_line(empty_data, max_widths, '+', '-')

    result = double_separator
    result += _make_line(header, max_widths)
    result += double_separator
    result += single_separator.join(
        _make_line(record, max_widths) for record in data)
    result += double_separator

    return result

  def _check_pipeline_existence(self,
                                pipeline_name: str,
                                required: bool = True) -> None:
    """Check if pipeline folder exists and if not, exit system.

    Args:
      pipeline_name: Name of the pipeline.
      required: Set it as True if pipeline needs to exist else set it to False.
    """
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name)
    # Check if pipeline folder exists.
    exists = fileio.exists(handler_pipeline_path)
    if required and not exists:
      # Check pipeline directory prior 0.25 and move files to the new location
      # automatically.
      old_handler_pipeline_path = os.path.join(
          self._get_deprecated_handler_home(), pipeline_name)
      if fileio.exists(old_handler_pipeline_path):
        fileio.makedirs(os.path.dirname(handler_pipeline_path))
        fileio.rename(old_handler_pipeline_path, handler_pipeline_path)
        engine_flag = self.flags_dict[labels.ENGINE_FLAG]
        handler_home_variable = engine_flag.upper() + '_HOME'
        click.echo(
            ('[WARNING] Pipeline "{pipeline_name}" was found in "{old_path}", '
             'but the location that TFX stores pipeline information was moved '
             'since TFX 0.25.0.\n'
             '[WARNING] Your files in "{old_path}" was automatically moved to '
             'the new location, "{new_path}".\n'
             '[WARNING] If you want to keep the files at the old location, set '
             '`{handler_home}` environment variable to "{old_handler_home}".'
            ).format(
                pipeline_name=pipeline_name,
                old_path=old_handler_pipeline_path,
                new_path=handler_pipeline_path,
                handler_home=handler_home_variable,
                old_handler_home=self._get_deprecated_handler_home()),
            err=True)
      else:
        sys.exit('Pipeline "{}" does not exist.'.format(pipeline_name))
    elif not required and exists:
      sys.exit('Pipeline "{}" already exists.'.format(pipeline_name))

  def _get_pipeline_info_path(self, pipeline_name):
    return os.path.join(self._handler_home_dir, pipeline_name)

  def _get_pipeline_args_path(self, pipeline_name):
    return os.path.join(
        self._get_pipeline_info_path(pipeline_name), 'pipeline_args.json')

  def get_schema(self):
    """Get the schema of the pipeline."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Path to pipeline args.
    pipeline_args_path = self._get_pipeline_args_path(pipeline_name)

    # Get pipeline_root.
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)

    self._read_schema_from_pipeline_root(pipeline_name,
                                         pipeline_args[labels.PIPELINE_ROOT])

  def _read_schema_from_pipeline_root(self, pipeline_name, pipeline_root):
    """Read Schema from the latest SchemaGen output of the pipeline."""
    # Check if pipeline root created. If not, it means that the user has not
    # created a run yet or the pipeline is still running for the first time.

    if not fileio.exists(pipeline_root):
      sys.exit(
          'Create a run before inferring schema. If pipeline is already running, then wait for it to successfully finish.'
      )

    # If pipeline_root exists, then check if SchemaGen output exists.
    components = fileio.listdir(pipeline_root)
    if 'SchemaGen' not in components:
      sys.exit(
          'Either SchemaGen component does not exist or pipeline is still running. If pipeline is running, then wait for it to successfully finish.'
      )

    # Get the latest SchemaGen output.
    component_output_dir = os.path.join(pipeline_root, 'SchemaGen')
    schema_dir = os.path.join(component_output_dir, 'schema')
    schemagen_outputs = fileio.listdir(schema_dir)
    latest_schema_folder = max(schemagen_outputs, key=int)

    # Copy schema to current dir.
    latest_schema_uri = base_driver._generate_output_uri(  # pylint: disable=protected-access
        component_output_dir, 'schema', int(latest_schema_folder))
    latest_schema_path = os.path.join(latest_schema_uri, 'schema.pbtxt')
    curr_dir_path = os.path.join(os.getcwd(), 'schema.pbtxt')
    io_utils.copy_file(latest_schema_path, curr_dir_path, overwrite=True)

    # Print schema and path to schema
    click.echo('Path to schema: {}'.format(curr_dir_path))
    click.echo('*********SCHEMA FOR {}**********'.format(pipeline_name.upper()))
    with open(curr_dir_path, 'r') as f:
      click.echo(f.read())
