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
"""Handler for Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

import click
import kfp
import kfp_server_api
from tabulate import tabulate
import tensorflow as tf

from typing import Text, Dict, Any
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


# TODO(b/132286477): Change generated api methods to client methods after SDK is
# updated.
class KubeflowHandler(base_handler.BaseHandler):
  """Helper methods for Kubeflow Handler."""

  def __init__(self, flags_dict: Dict[Text, Any]):
    """Initialize Kubeflow handler.

    Args:
      flags_dict: A dictionary with flags provided in a command.
    """
    super(KubeflowHandler, self).__init__(flags_dict)

    # TODO(b/132286477): Change to setup config instead of flags if needed.
    try:
      # Create client.
      self._client = kfp.Client(
          host=self.flags_dict[labels.ENDPOINT],
          client_id=self.flags_dict[labels.IAP_CLIENT_ID],
          namespace=self.flags_dict[labels.NAMESPACE])
    except kfp_server_api.rest.ApiException as err:
      self._print_error(err)

  def create_pipeline(self, overwrite: bool = False) -> None:
    """Creates pipeline in Kubeflow.

    Args:
      overwrite: set as true to update pipeline.
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
    """Updates pipeline in Kubeflow."""
    # Set overwrite as True to update the pipeline.
    self.create_pipeline(overwrite=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    try:
      response = self._client.list_pipelines()
    except kfp_server_api.rest.ApiException as err:
      self._print_error(err)

    if response.pipelines:
      click.echo(response.pipelines)
    else:
      click.echo('No pipelines to display.')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""
    try:
      pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
      # Check if pipeline exists on server.
      pipeline_id = self._get_pipeline_id(pipeline_name)
      self._client._pipelines_api.get_pipeline(pipeline_id)  # pylint: disable=protected-access

      # Delete pipeline for kfp server.
      self._client._pipelines_api.delete_pipeline(id=pipeline_id)  # pylint: disable=protected-access

      # Delete experiment from server.
      experiment_id = self._client.get_experiment(
          experiment_name=pipeline_name).id
      self._client._experiment_api.delete_experiment(experiment_id)  # pylint: disable=protected-access

    except kfp_server_api.rest.ApiException as err:
      sys.exit(self._print_error(err))

    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name,
                                         '')

    # Delete pipeline for home directory.
    io_utils.delete_dir(handler_pipeline_path)

    click.echo('Pipeline ' + pipeline_name + ' deleted successfully.')

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline in Kubeflow.

    Returns:
      pipeline_args: python dictionary with pipeline details extracted from DSL.
    """
    self._check_pipeline_dsl_path()
    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    self._check_pipeline_package_path()
    if not pipeline_args:
      sys.exit('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    click.echo('Pipeline package path: {}'.format(
        self.flags_dict[labels.PIPELINE_PACKAGE_PATH]))
    return pipeline_args

  def create_run(self) -> None:
    """Runs a pipeline in Kubeflow."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # Get pipeline id.
    pipeline_id = self._get_pipeline_id(pipeline_name)

    try:
      # Get experiment id.
      experiment_name = pipeline_name
      experiment_id = self._client.get_experiment(
          experiment_name=experiment_name).id

      # Run pipeline.
      run = self._client.run_pipeline(
          experiment_id=experiment_id,
          job_name=experiment_name,
          pipeline_id=pipeline_id)

    except kfp_server_api.rest.ApiException as err:
      sys.exit(self._print_error(err))

    click.echo('Run created for pipeline: ' + pipeline_name)
    self._print_runs([run])

  def delete_run(self) -> None:
    """Deletes a run."""
    try:
      self._client._run_api.delete_run(self.flags_dict[labels.RUN_ID])  # pylint: disable=protected-access
    except kfp_server_api.rest.ApiException as err:
      sys.exit(self._print_error(err))

    click.echo('Run deleted.')

  def terminate_run(self) -> None:
    """Stops a run."""
    try:
      self._client._run_api.terminate_run(self.flags_dict[labels.RUN_ID])  # pylint: disable=protected-access
    except kfp_server_api.rest.ApiException as err:
      sys.exit(self._print_error(err))

    click.echo('Run terminated.')

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    try:
      pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
      # Check if pipeline exists.
      pipeline_id = self._get_pipeline_id(pipeline_name)
      self._client._pipelines_api.get_pipeline(pipeline_id)  # pylint: disable=protected-access

      # Get experiment id.
      experiment_name = pipeline_name
      experiment_id = self._client.get_experiment(
          experiment_name=experiment_name).id

      # List runs.
      response = self._client.list_runs(experiment_id=experiment_id)

    except kfp_server_api.rest.ApiException as err:
      sys.exit(self._print_error(err))

    except ValueError as err:
      sys.exit(str(err))

    if response and response.runs:
      self._print_runs(response.runs)
    else:
      click.echo('No runs found.')

  def get_run(self) -> None:
    """Checks run status."""
    try:
      run = self._client.get_run(self.flags_dict[labels.RUN_ID]).run
      self._print_runs([run])
    except kfp_server_api.rest.ApiException as err:  # pylint: disable=broad-except
      sys.exit(self._print_error(err))

  def _save_pipeline(self, pipeline_args: Dict[Text, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory."""
    pipeline_name = pipeline_args[labels.PIPELINE_NAME]

    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name,
                                         '')

    # When updating pipeline delete pipeline from server and home dir.
    if tf.io.gfile.exists(handler_pipeline_path):

      # Delete pipeline for kfp server.
      pipeline_id = self._get_pipeline_id(pipeline_name)

      try:
        self._client._pipelines_api.delete_pipeline(id=pipeline_id)  # pylint: disable=protected-access
      except kfp_server_api.rest.ApiException as err:
        sys.exit(self._print_error(err))

      # Delete pipeline for home directory.
      io_utils.delete_dir(handler_pipeline_path)

    pipeline_package_path = self.flags_dict[labels.PIPELINE_PACKAGE_PATH]
    try:
      # Now upload pipeline to server.
      upload_response = self._client.upload_pipeline(
          pipeline_package_path=pipeline_package_path,
          pipeline_name=pipeline_name)
      click.echo(upload_response)

      # Create experiment with pipeline name as experiment name.
      experiment_name = pipeline_name
      experiment_id = self._client.create_experiment(experiment_name).id

    except kfp_server_api.rest.ApiException as err:
      sys.exit(self._print_error(err))

    # Add pipeline details to pipeline_args.
    pipeline_args[labels.PIPELINE_NAME] = upload_response.name
    pipeline_args[labels.PIPELINE_ID] = upload_response.id
    pipeline_args[labels.PIPELINE_PACKAGE_PATH] = pipeline_package_path
    pipeline_args[labels.EXPERIMENT_ID] = experiment_id

    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path,
                                      'pipeline_args.json')

    # Copy pipeline_args to pipeline folder.
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(pipeline_args_path, 'w') as f:
      json.dump(pipeline_args, f)

  def _check_pipeline_package_path(self):
    pipeline_package_path = self.flags_dict[labels.PIPELINE_PACKAGE_PATH]
    if not pipeline_package_path:
      sys.exit('Provide the output workflow package path.')

    if not tf.io.gfile.exists(pipeline_package_path):
      sys.exit('Pipeline package not found: {}'.format(pipeline_package_path))

  def _get_pipeline_id(self, pipeline_name: Text) -> Text:
    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name,
                                         '')

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path,
                                      'pipeline_args.json')
    # Get pipeline_id from pipeline_args.json
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)
    pipeline_id = pipeline_args[labels.PIPELINE_ID]
    return pipeline_id

  def _print_runs(self, runs):
    """Prints runs in a tabular format with headers mentioned below."""
    headers = ['pipeline_name', 'run_id', 'status', 'created_at']
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    data = [[pipeline_name, run.id, run.status,
             run.created_at.isoformat()] for run in runs]
    click.echo(tabulate(data, headers=headers, tablefmt='grid'))

  def _print_error(self, error: kfp_server_api.rest.ApiException):
    error = json.loads(error.body)
    click.echo('Error Code: {}'.format(str(error['code'])))
    click.echo('Error Type: {}'.format(error['details'][0]['@type']))
    click.echo('Error Message: {}'.format(error['message']))
