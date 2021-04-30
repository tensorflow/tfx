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
"""Handler for Kubeflow V2 runner."""

import functools
import os
import re
from typing import Any, Dict, Text

import click
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.handler import base_handler
from tfx.tools.cli.handler import kubeflow_handler
from tfx.tools.cli.kubeflow_v2.handler import kubeflow_v2_dag_runner_patcher
from tfx.utils import io_utils


_PIPELINE_ARG_FILE = 'pipeline_args.json'
_PIPELINE_SPEC_FILE = 'pipeline.json'

# Regex pattern used to capture the project id and the pipeline job name.
_FULL_JOB_NAME_PATTERN = r'projects/(\S+)/pipelineJobs/(\S+)'

# Prefix for the pipeline run detail page link.
_RUN_DETAIL_PREFIX = 'https://console.cloud.google.com/ai-platform/pipelines/runs/'


def _get_job_name(run: Dict[Text, Any]) -> Text:
  """Extracts the job name from its full name by regex.

  Args:
    run: JSON dict of a pipeline run object returned by Kubeflow pipelines REST
      API.

  Returns:
    Job name extracted from the given JSON dict.

  Raises:
    RuntimeError: if cannot find valid job name from the response.
  """
  full_name = run['name']
  match_result = re.match(_FULL_JOB_NAME_PATTERN, full_name)
  if not match_result:
    raise RuntimeError('Invalid job name is received.')
  return match_result.group(2)


def _get_job_link(job_name: Text, project_id: Text) -> Text:
  """Gets the link to the pipeline job UI according to job name and project."""
  return _RUN_DETAIL_PREFIX + '{job_name}?project={project_id}'.format(
      job_name=job_name, project_id=project_id)


class KubeflowV2Handler(base_handler.BaseHandler):
  """Helper methods for Kubeflow V2 Handler."""

  def __init__(self, flags_dict: Dict[Text, Any]):
    super().__init__(flags_dict)
    # Only when the given command is `run` and an API key is specified shall we
    # create a API client.
    # TODO(b/169095387): re-implement run commands when the unified client
    # becomes available.
    pass

  def create_pipeline(self, update: bool = False) -> None:
    """Creates or updates a pipeline to use in Kubeflow pipelines.

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
    """Updates pipeline in Kubeflow Pipelines."""
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

  def create_run(self) -> None:
    """Runs a pipeline in Kubeflow Pipelines."""
    # TODO(b/169095387): re-implement run commands when the unified client
    # becomes available.
    raise NotImplementedError('Creating a run has not been implemented for '
                              'Kubeflow V2 runner yet.')

  def terminate_run(self) -> None:
    """Stops a run."""
    # TODO(b/155096168): implement this.
    raise NotImplementedError('Terminating runs has not been implemented for '
                              'Kubeflow V2 runner yet.')

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    # TODO(b/169095387): re-implement run commands when the unified client
    # becomes available.
    raise NotImplementedError('Listing runs has not been implemented for '
                              'Kubeflow V2 runner yet.')

  def get_run(self) -> None:
    """Checks run status."""
    # TODO(b/169095387): re-implement run commands when the unified client
    # becomes available.
    raise NotImplementedError('Getting run status has not been implemented for '
                              'Kubeflow V2 runner yet.')

  def delete_run(self) -> None:
    """Deletes a run."""
    # TODO(b/155096168): implement this.
    raise NotImplementedError('Deleting runs has not been implemented for '
                              'Kubeflow V2 runner yet.')

  def _prepare_pipeline_dir(self, pipeline_name: str, required: bool) -> str:
    """Create a directory for pipeline definition in the handler directory."""

    self._check_pipeline_existence(pipeline_name, required)

    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name)

    # If updating pipeline, first delete the pipeline directory.
    if fileio.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    fileio.makedirs(handler_pipeline_path)

    # pipeline.json will be stored in KubeflowV2DagRunner.run().
    return handler_pipeline_path

