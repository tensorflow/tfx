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
"""Handler for Beam."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
from typing import Any, Dict, Text

import click

from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


class BeamHandler(base_handler.BaseHandler):
  """Helper methods for Beam Handler."""

  def create_pipeline(self, overwrite: bool = False) -> None:
    """Creates pipeline in Beam.

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
    """Updates pipeline in Beam."""

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
    """Deletes pipeline in Beam."""
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
    """Compiles pipeline in Beam.

    Returns:
      pipeline_args: python dictionary with pipeline details extracted from DSL.
    """
    self._check_pipeline_dsl_path()
    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    if not pipeline_args:
      sys.exit('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    return pipeline_args

  def create_run(self) -> None:
    """Runs a pipeline in Beam."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Get dsl path from pipeline args.
    pipeline_args_path = os.path.join(self._handler_home_dir, pipeline_name,
                                      'pipeline_args.json')
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)

    # Run pipeline dsl.
    self._subprocess_call(
        [sys.executable,
         str(pipeline_args[labels.PIPELINE_DSL_PATH])])

  def delete_run(self) -> None:
    """Deletes a run."""
    click.echo('Not supported for {} orchestrator.'.format(
        self.flags_dict[labels.ENGINE_FLAG]))

  def terminate_run(self) -> None:
    """Stops a run."""
    click.echo('Not supported for {} orchestrator.'.format(
        self.flags_dict[labels.ENGINE_FLAG]))

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    click.echo('Not supported for {} orchestrator.'.format(
        self.flags_dict[labels.ENGINE_FLAG]))

  def get_run(self) -> None:
    """Checks run status."""
    click.echo('Not supported for {} orchestrator.'.format(
        self.flags_dict[labels.ENGINE_FLAG]))

  def _save_pipeline(self, pipeline_args: Dict[Text, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory."""
    # Add pipeline dsl path to pipeline args.
    pipeline_args[labels.PIPELINE_DSL_PATH] = self.flags_dict[
        labels.PIPELINE_DSL_PATH]

    # Path to pipeline folder in beam.
    handler_pipeline_path = os.path.join(self._handler_home_dir,
                                         pipeline_args[labels.PIPELINE_NAME],
                                         '')

    # If updating pipeline, first delete pipeline directory.
    if fileio.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # Dump pipeline_args to handler pipeline folder as json.
    fileio.makedirs(handler_pipeline_path)
    with open(os.path.join(handler_pipeline_path, 'pipeline_args.json'),
              'w') as f:
      json.dump(pipeline_args, f)
