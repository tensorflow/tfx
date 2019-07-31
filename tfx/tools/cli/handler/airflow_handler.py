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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import subprocess
import sys
import click
import tensorflow as tf

from typing import Text, Dict, Any
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


class AirflowHandler(base_handler.BaseHandler):
  """Helper methods for Airflow Handler."""

  def __init__(self, flags_dict):
    super(AirflowHandler, self).__init__(flags_dict)
    self._handler_home_dir = self._get_handler_home('airflow')

  # TODO(b/132286477): Update comments after updating methods.
  # TODO(b/136201266): Update log messages similar to beam.
  def create_pipeline(self, overwrite: bool = False):
    """Creates pipeline in Airflow."""

    # Compile pipeline to check if pipeline_args are extracted successfully.
    pipeline_args = self.compile_pipeline()

    # Path to pipeline folder in airflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    if overwrite:
      # For update, check if pipeline exists.
      if not tf.io.gfile.exists(handler_pipeline_path):
        sys.exit('Pipeline {} does not exist.'
                 .format(pipeline_args[labels.PIPELINE_NAME]))
    else:
      # For create, verify that pipeline does not exist.
      if tf.io.gfile.exists(handler_pipeline_path):
        sys.exit('Pipeline {} already exists.'
                 .format(pipeline_args[labels.PIPELINE_NAME]))

    self._save_pipeline(pipeline_args)

  def update_pipeline(self):
    # Set overwrite to true for update to make sure pipeline exists.
    self.create_pipeline(overwrite=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    dags_folder = os.path.join(self._handler_home_dir, 'dags', '')
    if not tf.io.gfile.exists(dags_folder):
      click.echo('No pipelines to display.')
      return
    pipelines_list = tf.io.gfile.listdir(dags_folder)
    # Print every pipeline name in a new line.
    click.echo('\n'.join('{}' for _ in range(len(pipelines_list)))
               .format(*pipelines_list))

  def delete_pipeline(self) -> None:
    """Delete pipeline in Airflow."""
    # Path to pipeline folder in airflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        self.flags_dict[labels.PIPELINE_NAME])

    # Check if pipeline exists.
    if not tf.io.gfile.exists(handler_pipeline_path):
      sys.exit('Pipeline {} does not exist.'
               .format(self.flags_dict[labels.PIPELINE_NAME]))

    # Delete pipeline folder.
    io_utils.delete_dir(handler_pipeline_path)

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline in Airflow."""
    self._check_pipeline_dsl_path()
    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    if not pipeline_args:
      sys.exit('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    return pipeline_args

  def create_run(self) -> None:
    """Trigger DAG in Airflow."""
    # Check if pipeline exists.
    handler_pipeline_path = self._get_handler_pipeline_path(
        self.flags_dict[labels.PIPELINE_NAME])
    if not tf.io.gfile.exists(handler_pipeline_path):
      sys.exit('Pipeline {} does not exist.'
               .format(self.flags_dict[labels.PIPELINE_NAME]))
    # Unpause and trigger DAG.
    subprocess.call(['airflow', 'unpause',
                     self.flags_dict[labels.PIPELINE_NAME]])
    subprocess.call(
        ['airflow', 'trigger_dag', self.flags_dict[labels.PIPELINE_NAME]])

  def delete_run(self) -> None:
    """Deletes a run in Airflow."""
    click.echo('Not supported for Airflow.')

  def terminate_run(self) -> None:
    """Stops a run in Airflow."""
    click.echo('Not supported for Airflow.')

  def list_runs(self) -> None:
    """Lists all runs of a pipeline in Airflow."""
    # Check if pipeline exists.
    handler_pipeline_path = self._get_handler_pipeline_path(
        self.flags_dict[labels.PIPELINE_NAME])
    if not tf.io.gfile.exists(handler_pipeline_path):
      sys.exit('Pipeline {} does not exist.'.format(
          self.flags_dict[labels.PIPELINE_NAME]))
    subprocess.call(
        ['airflow', 'list_dag_runs', self.flags_dict[labels.PIPELINE_NAME]])

  def get_run(self) -> None:
    """Checks run status in Airflow."""
    # Check if pipeline exists.
    handler_pipeline_path = self._get_handler_pipeline_path(
        self.flags_dict[labels.PIPELINE_NAME])
    if not tf.io.gfile.exists(handler_pipeline_path):
      sys.exit('Pipeline {} does not exist.'.format(
          self.flags_dict[labels.PIPELINE_NAME]))
    dag_runs_list = str(
        subprocess.check_output(
            ['airflow', 'list_dag_runs',
             self.flags_dict[labels.PIPELINE_NAME]]))
    lines = dag_runs_list.split('\\n')
    for line in lines:
      # The tokens are id, run_id, state, execution_date, state_date
      tokens = line.split('|')
      if self.flags_dict[labels.RUN_ID] in line:
        click.echo('run_id :' + tokens[1])
        click.echo('state :' + tokens[2])

  def _save_pipeline(self, pipeline_args) -> None:
    """Creates/updates pipeline folder in the handler directory."""

    # Path to pipeline folder in airflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    # If updating pipeline, first delete pipeline directory.
    if tf.io.gfile.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # Dump pipeline_args to handler pipeline folder as json.
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(os.path.join(
        handler_pipeline_path, 'pipeline_args.json'), 'w') as f:
      json.dump(pipeline_args, f)

    # Copy dsl to pipeline folder
    io_utils.copy_file(
        self.flags_dict[labels.PIPELINE_DSL_PATH],
        os.path.join(
            handler_pipeline_path,
            os.path.basename(self.flags_dict[labels.PIPELINE_DSL_PATH])
            )
        )

  def _get_handler_pipeline_path(self, pipeline_name) -> Text:
    """Path to pipeline folder in airflow.

    Args:
      pipeline_name: name of the pipeline

    Returns:
      Path to pipeline folder in airflow.
    """
    # Path to pipeline folder in airflow.
    return os.path.join(self._handler_home_dir, 'dags', pipeline_name)
