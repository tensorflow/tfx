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
"""Handler for Vertex runner."""

import functools
import os
import sys
import click

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.tools.cli.handler import kubeflow_handler
from tfx.tools.cli.handler import kubeflow_v2_dag_runner_patcher
from tfx.utils import io_utils


class VertexHandler(base_handler.BaseHandler):
  """Helper methods for Vertex Handler."""

  def create_pipeline(self, update: bool = False) -> None:
    """Creates or updates a pipeline to use in Vertex Pipelines.

    Args:
      update: set as true to update pipeline.
    """
    if self.flags_dict.get(labels.BUILD_IMAGE):
      build_image_fn = functools.partial(
          kubeflow_handler.create_container_image,
          base_image=self.flags_dict.get(labels.BASE_IMAGE))
    else:
      build_image_fn = None

    patcher = kubeflow_v2_dag_runner_patcher.KubeflowV2DagRunnerPatcher(
        call_real_run=True,
        build_image_fn=build_image_fn,
        prepare_dir_fn=functools.partial(
            self._prepare_pipeline_dir, required=update))
    context = self.execute_dsl(patcher)
    pipeline_name = context[patcher.PIPELINE_NAME]

    if update:
      click.echo('Pipeline "{}" updated successfully.'.format(pipeline_name))
    else:
      click.echo('Pipeline "{}" created successfully.'.format(pipeline_name))

  def update_pipeline(self) -> None:
    """Updates pipeline in Vertex Pipelines."""
    self.create_pipeline(update=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    # There is no managed storage for pipeline packages, so CLI consults
    # local dir to list pipelines.
    if not fileio.exists(self._handler_home_dir):
      click.echo('No pipelines to display.')
      return
    pipelines_list = fileio.listdir(self._handler_home_dir)

    # Print every pipeline name in a new line.
    click.echo('-' * 30)
    click.echo('\n'.join(pipelines_list))
    click.echo('-' * 30)

  def delete_pipeline(self) -> None:
    """Delete pipeline in the environment."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    self._check_pipeline_existence(pipeline_name)

    io_utils.delete_dir(os.path.join(self._handler_home_dir, pipeline_name))

    click.echo('Pipeline ' + pipeline_name + ' deleted successfully.')

  def compile_pipeline(self) -> None:
    """Compiles pipeline into Kubeflow V2 Pipelines spec."""
    patcher = kubeflow_v2_dag_runner_patcher.KubeflowV2DagRunnerPatcher(
        call_real_run=True)
    context = self.execute_dsl(patcher)
    click.echo(f'Pipeline {context[patcher.PIPELINE_NAME]} compiled '
               'successfully.')

  def _prepare_vertex(self) -> None:
    if not self.flags_dict[labels.GCP_PROJECT_ID]:
      sys.exit('Please set GCP project id with --project flag.')
    if not self.flags_dict[labels.GCP_REGION]:
      sys.exit('Please set GCP region with --region flag.')

    aiplatform.init(
        project=self.flags_dict[labels.GCP_PROJECT_ID],
        location=self.flags_dict[labels.GCP_REGION],
    )

  def create_run(self) -> None:
    """Runs a pipeline in Vertex Pipelines."""
    self._prepare_vertex()
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    # In Vertex AI, runtime parameter string value is parsed from the server,
    # so client directly sends Dict[str, str] value.
    unparsed_runtime_parameters = self.flags_dict[labels.RUNTIME_PARAMETER]

    job = pipeline_jobs.PipelineJob(
        display_name=pipeline_name,
        template_path=self._get_pipeline_definition_path(pipeline_name),
        parameter_values=unparsed_runtime_parameters)
    # TODO(b/198114641): Delete pytype exception after upgrading source code
    # to aiplatform>=1.3.

    job.run(sync=False)  # pytype: disable=attribute-error
    job.wait_for_resource_creation()  # pytype: disable=attribute-error

    click.echo('Run created for pipeline: ' + pipeline_name)
    self._print_run(job)

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
    self._prepare_vertex()
    job = pipeline_jobs.PipelineJob.get(  # pytype: disable=attribute-error
        resource_name=self.flags_dict[labels.RUN_ID])
    self._print_run(job)

  def delete_run(self) -> None:
    """Deletes a run."""
    click.echo('Not supported for {} orchestrator.'.format(
        self.flags_dict[labels.ENGINE_FLAG]))

  def _get_pipeline_dir(self, pipeline_name: str) -> str:
    return os.path.join(self._handler_home_dir, pipeline_name)

  def _get_pipeline_definition_path(self, pipeline_name: str) -> str:
    return os.path.join(
        self._get_pipeline_dir(pipeline_name),
        kubeflow_v2_dag_runner_patcher.OUTPUT_FILENAME)

  def _prepare_pipeline_dir(self, pipeline_name: str, required: bool) -> str:
    """Create a directory for pipeline definition in the handler directory."""

    self._check_pipeline_existence(pipeline_name, required)

    handler_pipeline_path = self._get_pipeline_dir(pipeline_name)

    # If updating pipeline, first delete the pipeline directory.
    if fileio.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    fileio.makedirs(handler_pipeline_path)

    # pipeline.json will be stored in KubeflowV2DagRunner.run().
    return handler_pipeline_path

  def _print_run(self, run):
    """Prints a run in a tabular format with headers mentioned below."""
    headers = ['run_id', 'status', 'created_at', 'link']
    data = [[
        run.name, run.state.name, run.create_time,
        run._dashboard_uri()  # pylint: disable=protected-access
    ]]
    click.echo(self._format_table(headers, data))
