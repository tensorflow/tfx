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

import json
import sys
from typing import Any, Dict

import click

from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.tools.cli.handler import beam_dag_runner_patcher
from tfx.tools.cli.handler import dag_runner_patcher
from tfx.utils import io_utils


class BeamHandler(base_handler.BaseHandler):
  """Helper methods for Beam Handler."""

  def _get_dag_runner_patcher(self) -> dag_runner_patcher.DagRunnerPatcher:
    return beam_dag_runner_patcher.BeamDagRunnerPatcher()

  def create_pipeline(self, overwrite: bool = False) -> None:
    """Creates pipeline in Beam.

    Args:
      overwrite: set as true to update pipeline.
    """
    patcher = self._get_dag_runner_patcher()
    context = self.execute_dsl(patcher)
    pipeline_name = context[patcher.PIPELINE_NAME]

    self._check_pipeline_existence(pipeline_name, required=overwrite)
    self._save_pipeline({
        labels.PIPELINE_NAME: pipeline_name,
        labels.PIPELINE_ROOT: context[patcher.PIPELINE_ROOT]
    })

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
    handler_pipeline_path = self._get_pipeline_info_path(pipeline_name)

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Delete pipeline folder.
    io_utils.delete_dir(handler_pipeline_path)
    click.echo('Pipeline "{}" deleted successfully.'.format(pipeline_name))

  def compile_pipeline(self) -> None:
    """Compiles pipeline in Beam."""
    patcher = self._get_dag_runner_patcher()
    self.execute_dsl(patcher)
    click.echo('Pipeline compiled successfully.')

  def create_run(self) -> None:
    """Runs a pipeline in Beam."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    with open(self._get_pipeline_args_path(pipeline_name), 'r') as f:
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

  def _save_pipeline(self, pipeline_args: Dict[str, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory."""
    # Add pipeline dsl path to pipeline args.
    pipeline_args[labels.PIPELINE_DSL_PATH] = self.flags_dict[
        labels.PIPELINE_DSL_PATH]
    pipeline_name = pipeline_args[labels.PIPELINE_NAME]

    handler_pipeline_path = self._get_pipeline_info_path(pipeline_name)

    # If updating pipeline, first delete pipeline directory.
    if fileio.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # Dump pipeline_args to handler pipeline folder as json.
    fileio.makedirs(handler_pipeline_path)
    with open(self._get_pipeline_args_path(pipeline_name),
              'w') as f:
      json.dump(pipeline_args, f)
