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
"""Handler for Beam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
import click
import tensorflow as tf
from typing import Dict, Text, Any

from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


class BeamHandler(base_handler.BaseHandler):
  """Helper methods for Beam Handler."""

  def __init__(self, flags_dict: Dict[Text, Any]):
    super(BeamHandler, self).__init__(flags_dict)
    self._handler_home_dir = self._get_handler_home('beam')

  def create_pipeline(self, overwrite: bool = False):
    """Creates pipeline in Beam."""

    # Compile pipeline to check if pipeline_args are extracted successfully.
    pipeline_args = self.compile_pipeline()

    # Path to pipeline folder in airflow.
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

  def update_pipeline(self) -> None:
    """Updates pipeline in Beam."""
    # Set overwrite to true for update to make sure pipeline exists.
    self.create_pipeline(overwrite=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    if not tf.io.gfile.exists(self._handler_home_dir):
      click.echo('No pipelines to display.')
      return
    pipelines_list = tf.io.gfile.listdir(self._handler_home_dir)

    # Print every pipeline name in a new line.
    click.echo('-' * 30)
    click.echo('\n'.join(pipelines_list))
    click.echo('-' * 30)

  def delete_pipeline(self) -> None:
    """Deletes pipeline in Beam."""
    # Path to pipeline folder.
    handler_pipeline_path = self._get_handler_pipeline_path(
        self.flags_dict[labels.PIPELINE_NAME])

    # Check if pipeline exists.
    if not tf.io.gfile.exists(handler_pipeline_path):
      sys.exit('Pipeline {} does not exist.'.format(
          self.flags_dict[labels.PIPELINE_NAME]))

    # Delete pipeline folder.
    io_utils.delete_dir(handler_pipeline_path)

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline in Beam."""
    self._check_pipeline_dsl_path()
    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    if not pipeline_args:
      sys.exit('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    return pipeline_args

  def create_run(self) -> None:
    """Runs a pipeline in Beam."""
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

  def _save_pipeline(self, pipeline_args: Dict[Text, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory."""

    # Add pipeline dsl path to pipeline args.
    pipeline_args[labels.PIPELINE_DSL_PATH] = self.flags_dict[
        labels.PIPELINE_DSL_PATH]

    # Path to pipeline folder in beam.
    handler_pipeline_path = self._get_handler_pipeline_path(
        pipeline_args[labels.PIPELINE_NAME])

    # If updating pipeline, first delete pipeline directory.
    if tf.io.gfile.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # Dump pipeline_args to handler pipeline folder as json.
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(os.path.join(handler_pipeline_path, 'pipeline_args.json'),
              'w') as f:
      json.dump(pipeline_args, f)

  def _get_handler_pipeline_path(self, pipeline_name: Text) -> Text:
    """Path to pipeline folder in beam.

    Args:
      pipeline_name: name of the pipeline

    Returns:
      Path to pipeline folder in beam.
    """
    # Path to pipeline folder in beam.
    return os.path.join(self._handler_home_dir, pipeline_name)
