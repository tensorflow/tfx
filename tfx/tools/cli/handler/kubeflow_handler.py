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
"""Handler for Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Text

import click
import kfp
from tabulate import tabulate
import tensorflow.compat.v1 as tf

from tfx.tools.cli import labels
from tfx.tools.cli.container_builder import builder
from tfx.tools.cli.container_builder import labels as container_builder_labels
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
    if labels.NAMESPACE in self.flags_dict:
      self._client = kfp.Client(
          host=self.flags_dict[labels.ENDPOINT],
          client_id=self.flags_dict[labels.IAP_CLIENT_ID],
          namespace=self.flags_dict[labels.NAMESPACE])
    else:
      self._client = None

  def create_pipeline(self, overwrite: bool = False) -> None:
    """Creates pipeline in Kubeflow.

    Args:
      overwrite: set as true to update pipeline.
    """
    # Build pipeline container image.
    try:
      target_image = self.flags_dict.get(labels.TARGET_IMAGE)
      skaffold_cmd = self.flags_dict.get(labels.SKAFFOLD_CMD)
      if target_image is not None or os.path.exists(
          container_builder_labels.BUILD_SPEC_FILENAME):
        base_image = self.flags_dict.get(labels.BASE_IMAGE)
        target_image = self._build_pipeline_image(target_image, base_image,
                                                  skaffold_cmd)
        os.environ[labels.KUBEFLOW_TFX_IMAGE_ENV] = target_image
    except (ValueError, subprocess.CalledProcessError, RuntimeError):
      click.echo('No container image is built.')
      raise
    else:
      click.echo('New container image is built. Target image is available in '
                 'the build spec file.')

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
    # TODO(jyzhao): use metadata context to get the pipeline info.
    response = self._client.list_pipelines(page_size=100)

    if response and response.pipelines:
      click.echo(response.pipelines)
    else:
      click.echo('No pipelines to display.')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""

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
    self._check_pipeline_package_path(pipeline_args[labels.PIPELINE_NAME])
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

    # Get experiment id.
    experiment_name = pipeline_name
    experiment_id = self._client.get_experiment(
        experiment_name=experiment_name).id

    # Run pipeline.
    run = self._client.run_pipeline(
        experiment_id=experiment_id,
        job_name=experiment_name,
        pipeline_id=pipeline_id)

    click.echo('Run created for pipeline: ' + pipeline_name)
    self._print_runs([run])

  def delete_run(self) -> None:
    """Deletes a run."""
    self._client._run_api.delete_run(self.flags_dict[labels.RUN_ID])  # pylint: disable=protected-access

    click.echo('Run deleted.')

  def terminate_run(self) -> None:
    """Stops a run."""
    self._client._run_api.terminate_run(self.flags_dict[labels.RUN_ID])  # pylint: disable=protected-access
    click.echo('Run terminated.')

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    # Check if pipeline exists.
    pipeline_id = self._get_pipeline_id(pipeline_name)
    self._client._pipelines_api.get_pipeline(pipeline_id)  # pylint: disable=protected-access

    # Get experiment id.
    experiment_name = pipeline_name
    experiment_id = self._client.get_experiment(
        experiment_name=experiment_name).id

    # List runs.
    # TODO(jyzhao): use metadata context to get the run info.
    response = self._client.list_runs(experiment_id=experiment_id)

    if response and response.runs:
      self._print_runs(response.runs)
    else:
      click.echo('No runs found.')

  def get_run(self) -> None:
    """Checks run status."""

    run = self._client.get_run(self.flags_dict[labels.RUN_ID]).run
    self._print_runs([run])

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

      self._client._pipelines_api.delete_pipeline(id=pipeline_id)  # pylint: disable=protected-access

      # Delete pipeline for home directory.
      io_utils.delete_dir(handler_pipeline_path)

    pipeline_package_path = self.flags_dict[labels.PIPELINE_PACKAGE_PATH]

    # Now upload pipeline to server.
    upload_response = self._client.upload_pipeline(
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name)
    click.echo(upload_response)

    # Create experiment with pipeline name as experiment name.
    experiment_name = pipeline_name
    experiment_id = self._client.create_experiment(experiment_name).id

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

  def _check_pipeline_package_path(self, pipeline_name: Text) -> None:

    # When unset, search for the workflow file in the current dir.
    if not self.flags_dict[labels.PIPELINE_PACKAGE_PATH]:
      self.flags_dict[labels.PIPELINE_PACKAGE_PATH] = os.path.join(
          os.getcwd(), '{}.tar.gz'.format(pipeline_name))

    pipeline_package_path = self.flags_dict[labels.PIPELINE_PACKAGE_PATH]
    if not tf.io.gfile.exists(pipeline_package_path):
      sys.exit(
          'Pipeline package not found at {}. When --package_path is unset, it will try to find the workflow file, "<pipeline_name>.tar.gz" in the current directory.'
          .format(pipeline_package_path))

  def _build_pipeline_image(self,
                            target_image: Optional[Text] = None,
                            base_image: Optional[Text] = None,
                            skaffold_cmd: Optional[Text] = None) -> None:
    return builder.ContainerBuilder(
        target_image=target_image,
        base_image=base_image,
        skaffold_cmd=skaffold_cmd).build()

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
