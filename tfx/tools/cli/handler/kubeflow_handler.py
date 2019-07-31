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
import tensorflow as tf

from typing import Text, Dict, Any
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


# TODO(b/132286477): Change generated api methods to client methods after SDK is
# updated.
class KubeflowHandler(base_handler.BaseHandler):
  """Helper methods for Kubeflow Handler."""

  # TODO(b/132286477): Update comments after updating methods.

  def __init__(self, flags_dict: Dict[Text, Any]):
    self.flags_dict = flags_dict
    self._handler_home_dir = self._get_handler_home('kubeflow_pipelines')

    # TODO(b/132286477): Change to setup config instead of flags if needed.
    # Create client.
    self._client = kfp.Client(
        host=self.flags_dict[labels.ENDPOINT],
        client_id=self.flags_dict[labels.IAP_CLIENT_ID],
        namespace=self.flags_dict[labels.NAMESPACE])

  def create_pipeline(self, overwrite: bool = False) -> None:
    """Creates pipeline in Kubeflow."""

    # Compile pipeline to check if pipeline_args are extracted successfully.
    pipeline_args = self.compile_pipeline()

    # Path to pipeline folder in kubeflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    if overwrite:
      # For update, check if pipeline exists.
      if not tf.io.gfile.exists(handler_pipeline_path):
        sys.exit('Pipeline {} does not exist.'.format(
            pipeline_args[labels.PIPELINE_NAME]))
    else:
      # For create, verify that pipeline does not exist.
      if tf.io.gfile.exists(handler_pipeline_path):
        sys.exit('Pipeline {} already exists.'.format(
            pipeline_args[labels.PIPELINE_NAME]))

    self._save_pipeline(pipeline_args)

    if overwrite:
      click.echo('Pipeline {} updated successfully.'.format(
          pipeline_args[labels.PIPELINE_NAME]))
    else:
      click.echo('Pipeline {} created successfully.'.format(
          pipeline_args[labels.PIPELINE_NAME]))

  def update_pipeline(self) -> None:
    """Updates pipeline in Kubeflow."""
    # Set overwrite to true for update to make sure pipeline exists.
    self.create_pipeline(overwrite=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    response = self._client.list_pipelines()
    if response.pipelines:
      click.echo(response.pipelines)
    else:
      click.echo('No pipelines to display.')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""
    # Check if pipeline exists.
    try:
      pipeline_id = self._get_pipeline_id(self.flags_dict[labels.PIPELINE_NAME])
      self._client._pipelines_api.get_pipeline(pipeline_id)  # pylint: disable=protected-access
    except (IOError, RuntimeError):
      sys.exit('Pipeline {} does not exist.'.format(
          self.flags_dict[labels.PIPELINE_NAME]))

    # Path to pipeline folder.
    handler_pipeline_path = self._get_handler_pipeline_path(
        self.flags_dict[labels.PIPELINE_NAME])

    # Delete pipeline for kfp server.
    pipeline_id = self._get_pipeline_id(self.flags_dict[labels.PIPELINE_NAME])
    self._client._pipelines_api.delete_pipeline(id=pipeline_id)  # pylint: disable=protected-access

    # Delete pipeline for home directory.
    io_utils.delete_dir(handler_pipeline_path)

    click.echo('Pipeline ' + self.flags_dict[labels.PIPELINE_NAME] +
               ' deleted successfully.')

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline in Kubeflow."""
    self._check_pipeline_dsl_path()
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
    pass

  def delete_run(self) -> None:
    """Deletes a run."""
    pass

  def terminate_run(self) -> None:
    """Stops a run."""
    pass

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    pass

  def get_run(self) -> None:
    """Checks run status."""
    pass

  def _get_handler_pipeline_path(self, pipeline_name: Text) -> Text:
    """Path to pipeline folder in Kubeflow.

    Args:
      pipeline_name: name of the pipeline

    Returns:
      Path to pipeline folder in Kubeflow.
    """
    # Path to pipeline folder in Kubeflow.
    return os.path.join(self._handler_home_dir, pipeline_name, '')

  def _save_pipeline(self, pipeline_args: Dict[Text, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory."""

    # Path to pipeline folder in Kubeflow.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    # When updating pipeline delete pipeline from server and home dir.
    if tf.io.gfile.exists(handler_pipeline_path):

      # Delete pipeline for kfp server.
      pipeline_id = self._get_pipeline_id(pipeline_args[labels.PIPELINE_NAME])
      self._client._pipelines_api.delete_pipeline(id=pipeline_id)  # pylint: disable=protected-access

      # Delete pipeline for home directory.
      io_utils.delete_dir(handler_pipeline_path)

      # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path,
                                      'pipeline_args.json')

    # Now upload pipeline to server.
    upload_response = self._client.upload_pipeline(
        pipeline_package_path=self.flags_dict[labels.PIPELINE_PACKAGE_PATH],
        pipeline_name=pipeline_args[labels.PIPELINE_NAME])
    click.echo(upload_response)

    # Add pipeline_id and pipeline_name to pipeline_args.
    pipeline_args[labels.PIPELINE_NAME] = upload_response.name
    pipeline_args[labels.PIPELINE_ID] = upload_response.id

    # Copy pipeline_args to pipeline folder.
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(pipeline_args_path, 'w') as f:
      json.dump(pipeline_args, f)

  def _check_pipeline_package_path(self):
    if not self.flags_dict[labels.PIPELINE_PACKAGE_PATH]:
      sys.exit('Provide the output workflow package path.')

    if not tf.io.gfile.exists(self.flags_dict[labels.PIPELINE_PACKAGE_PATH]):
      sys.exit('Pipeline package not found: {}'.format(
          self.flags_dict[labels.PIPELINE_PACKAGE_PATH]))

  def _get_pipeline_id(self, pipeline_name: Text) -> Text:
    # Path to pipeline folder in Kubeflow.
    handler_pipeline_path = self._get_handler_pipeline_path(pipeline_name)

    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path,
                                      'pipeline_args.json')
    # Get pipeline_id from pipeline_args.json
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)
    pipeline_id = pipeline_args[labels.PIPELINE_ID]
    return pipeline_id
