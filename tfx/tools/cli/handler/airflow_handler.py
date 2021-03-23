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
"""Handler for Airflow."""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Text

import click

from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


class AirflowHandler(base_handler.BaseHandler):
  """Helper methods for Airflow Handler."""

  def __init__(self, flags_dict):
    self._check_airflow_version()
    super(AirflowHandler, self).__init__(flags_dict)
    self._handler_home_dir = os.path.join(self._handler_home_dir, 'dags', '')

  def _get_airflow_version(self):
    return subprocess.check_output(['airflow', 'version'],
                                   encoding='utf-8').rstrip()

  def _check_airflow_version(self):
    [major, minor, patch] = self._get_airflow_version().split('.')
    if (int(major), int(minor), int(patch)) < (1, 10, 14):
      raise RuntimeError('Apache-airflow 1.10.14 or later required.')

  def create_pipeline(self, overwrite: bool = False) -> None:
    """Creates pipeline in Airflow.

    Args:
      overwrite: Set as True to update pipeline.
    """
    # Compile pipeline to check if pipeline_args are extracted successfully.
    pipeline_args = self.compile_pipeline()

    pipeline_name = pipeline_args[labels.PIPELINE_NAME]

    self._check_pipeline_existence(pipeline_name, required=overwrite)

    self._save_pipeline(pipeline_args)

    if overwrite:
      click.echo('Pipeline "{}" updated successfully.'.format(pipeline_name))
    else:
      click.echo('Pipeline "{}" created successfully.'.format(pipeline_name))

  def update_pipeline(self) -> None:
    """Updates pipeline in Airflow."""
    # Set overwrite as True to update the pipeline.
    self.create_pipeline(overwrite=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    if not fileio.exists(self._handler_home_dir):
      click.echo('No pipelines to display.')
      return

    pipelines_list = fileio.listdir(self._handler_home_dir)

    # Print every pipeline name in a new line.
    click.echo('-' * 30)
    click.echo('\n'.join(pipelines_list))
    click.echo('-' * 30)

  def delete_pipeline(self) -> None:
    """Delete pipeline in Airflow."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name,
                                         '')

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Delete pipeline folder.
    io_utils.delete_dir(handler_pipeline_path)
    click.echo('Pipeline "{}" deleted successfully.'.format(pipeline_name))

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline in Airflow.

    Returns:
      A python dictionary with pipeline details extracted from DSL.
    """
    self._check_pipeline_dsl_path()
    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    if not pipeline_args:
      sys.exit('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    return pipeline_args

  def create_run(self) -> None:
    """Trigger DAG in Airflow."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Unpause DAG.
    self._subprocess_call(['airflow', 'dags', 'unpause', pipeline_name])

    # Trigger DAG.
    self._subprocess_call(['airflow', 'dags', 'trigger', pipeline_name])

    click.echo('Run created for pipeline: ' + pipeline_name)

  def delete_run(self) -> None:
    """Deletes a run in Airflow."""
    click.echo('Not supported for Airflow.')

  def terminate_run(self) -> None:
    """Stops a run in Airflow."""
    click.echo('Not supported for Airflow.')

  def _get_all_runs(self, pipeline_name) -> List[Dict[str, str]]:
    dag_runs_list = subprocess.check_output(
        ['airflow', 'dags', 'list-runs', '-d', pipeline_name, '-o',
         'json'], encoding='utf-8')
    return json.loads(dag_runs_list)

  def _print_runs(self, runs):
    """Prints runs in a tabular format with headers mentioned below."""
    headers = ('pipeline_name', 'run_id', 'state', 'execution_date',
               'start_date', 'end_date')
    data = []
    for run in runs:
      # Replace header name 'dag_id' with 'pipeline_name'.
      data.append([run['dag_id'], *[run[key] for key in headers[1:]]])
    click.echo(self._format_table(headers, data))

  def list_runs(self) -> None:
    """Lists all runs of a pipeline in Airflow."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    self._check_pipeline_existence(pipeline_name)
    runs = self._get_all_runs(pipeline_name)
    self._print_runs(runs)

  def get_run(self) -> None:
    """Checks run status in Airflow."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    run_id = self.flags_dict[labels.RUN_ID]
    self._check_pipeline_existence(pipeline_name)

    runs = self._get_all_runs(pipeline_name)
    for run in runs:
      if run['run_id'] == run_id:
        self._print_runs([run])
    else:
      click.echo(
          f'Run "{run_id}" of pipeline "{pipeline_name}" does not exist.')

  def _save_pipeline(self, pipeline_args: Dict[Text, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory.

    Args:
      pipeline_args: Pipeline details obtained from DSL.
    """
    # Path to pipeline folder in Airflow.
    handler_pipeline_path = os.path.join(self._handler_home_dir,
                                         pipeline_args[labels.PIPELINE_NAME],
                                         '')

    # If updating pipeline, first delete pipeline directory.
    if fileio.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # Dump pipeline_args to handler pipeline folder as json.
    fileio.makedirs(handler_pipeline_path)
    with open(os.path.join(
        handler_pipeline_path, 'pipeline_args.json'), 'w') as f:
      json.dump(pipeline_args, f)

    # Copy dsl to pipeline folder
    pipeline_dsl_path = self.flags_dict[labels.PIPELINE_DSL_PATH]
    io_utils.copy_file(
        pipeline_dsl_path,
        os.path.join(handler_pipeline_path,
                     os.path.basename(pipeline_dsl_path)))
