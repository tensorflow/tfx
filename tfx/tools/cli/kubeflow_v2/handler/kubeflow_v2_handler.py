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

import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Text

import click
from tabulate import tabulate
from tfx.dsl.io import fileio
from tfx.tools.cli import labels
from tfx.tools.cli.container_builder import builder
from tfx.tools.cli.container_builder import labels as container_builder_labels
from tfx.tools.cli.handler import base_handler
from tfx.tools.cli.kubeflow_v2 import labels as kubeflow_labels
from tfx.utils import io_utils

from google.protobuf import json_format

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
    # Build pipeline container image.
    try:
      target_image = self.flags_dict.get(kubeflow_labels.TFX_IMAGE_ENV)
      skaffold_cmd = self.flags_dict.get(labels.SKAFFOLD_CMD)
      if target_image is not None or os.path.exists(
          container_builder_labels.BUILD_SPEC_FILENAME):
        base_image = self.flags_dict.get(labels.BASE_IMAGE)
        target_image = self._build_pipeline_image(target_image, base_image,
                                                  skaffold_cmd)
        os.environ[kubeflow_labels.TFX_IMAGE_ENV] = target_image
    except (ValueError, subprocess.CalledProcessError, RuntimeError):
      click.echo('No container image is built.')
      raise
    else:
      click.echo('New container image is built. Target image is available in '
                 'the build spec file.')

    # Compile pipeline to check if pipeline_args are extracted successfully.
    pipeline_args = self.compile_pipeline()

    pipeline_name = pipeline_args[labels.PIPELINE_NAME]

    self._check_pipeline_existence(pipeline_name, required=update)

    self._save_pipeline(pipeline_args)

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

    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name,
                                         '')
    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # Delete pipeline for home directory.
    io_utils.delete_dir(handler_pipeline_path)

    click.echo('Pipeline ' + pipeline_name + ' deleted successfully.')

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline into Kubeflow Pipelines spec.

    Returns:
      pipeline_args: python dictionary with pipeline details extracted from DSL.

    Raises:
      RuntimeError: when pipeline dsl compilation fails.
    """
    # TODO(b/155096168): implement this by actually invoking .compile() method.
    # Currently it directly assigns the pipeline_args with the pipeline DSL path
    # and the pipeline name.
    self._check_pipeline_dsl_path()

    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    if not pipeline_args:
      raise RuntimeError('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    return pipeline_args

  def _extract_pipeline_args(self) -> Dict[Text, Any]:
    """Get pipeline args from the DSL by compiling the pipeline.

    Returns:
      Python dictionary with pipeline details extracted from DSL.

    Raises:
      RuntimeError: when the given pipeline arg file location is occupied.
    """
    pipeline_dsl_path = self.flags_dict[labels.PIPELINE_DSL_PATH]

    if os.path.isdir(pipeline_dsl_path):
      sys.exit('Provide a valid dsl file path.')

    # Create an environment for subprocess.
    temp_env = os.environ.copy()

    # We don't need image name and project ID for extracting pipeline info,
    # so they can be optional.
    runner_env = {
        kubeflow_labels.TFX_IMAGE_ENV:
            self.flags_dict.get(kubeflow_labels.TFX_IMAGE_ENV, ''),
        kubeflow_labels.GCP_PROJECT_ID_ENV:
            self.flags_dict.get(kubeflow_labels.GCP_PROJECT_ID_ENV, ''),
    }

    temp_env.update(runner_env)

    # Run pipeline dsl. Note that here because we don't have RUN_FLAG_ENV
    # the actual execution won't be triggered. Instead the DSL will output a
    # compiled pipeline spec.
    self._subprocess_call(
        command=[sys.executable, pipeline_dsl_path], env=temp_env)

    # Only import pipeline_pb2 when needed to guard CLI dependency.
    from tfx.orchestration.kubeflow.v2.proto import pipeline_pb2  # pylint: disable=g-import-not-at-top

    # Extract the needed information from compiled pipeline spec.
    job_message = pipeline_pb2.PipelineJob()
    io_utils.parse_json_file(
        file_name=os.path.join(os.getcwd(), _PIPELINE_SPEC_FILE),
        message=job_message)

    pipeline_spec_pb = json_format.ParseDict(job_message.pipeline_spec,
                                             pipeline_pb2.PipelineSpec())

    pipeline_name = pipeline_spec_pb.pipeline_info.name
    pipeline_args = {
        'pipeline_name': pipeline_name,
        'pipeline_root': job_message.runtime_config.gcs_output_directory
    }

    return pipeline_args

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

  # TODO(b/156746891) merge _check_dsl_runner() back to base_handler for
  # consistency.
  def _check_dsl_runner(self) -> None:
    """Checks if runner in dsl is Kubeflow V2 runner."""
    with open(self.flags_dict[labels.PIPELINE_DSL_PATH], 'r') as f:
      dsl_contents = f.read()
      if 'KubeflowV2DagRunner' not in dsl_contents:
        raise RuntimeError('KubeflowV2DagRunner not found in dsl.')

  def _save_pipeline(self, pipeline_args: Dict[Text, Any]) -> None:
    """Creates/updates pipeline folder in the handler directory."""
    # Add pipeline dsl path to pipeline args.
    pipeline_args[labels.PIPELINE_DSL_PATH] = self.flags_dict[
        labels.PIPELINE_DSL_PATH]

    # Path to pipeline folder in beam.
    handler_pipeline_path = os.path.join(self._handler_home_dir,
                                         pipeline_args[labels.PIPELINE_NAME])

    # If updating pipeline, first delete the pipeline directory.
    if fileio.exists(handler_pipeline_path):
      io_utils.delete_dir(handler_pipeline_path)

    # TODO(b/157599419): Consider deprecating PipelineArgs.
    # Dump pipeline_args to handler pipeline folder as json.
    fileio.makedirs(handler_pipeline_path)
    with open(os.path.join(handler_pipeline_path, _PIPELINE_ARG_FILE),
              'w') as f:
      json.dump(pipeline_args, f)

  def _build_pipeline_image(self,
                            target_image: Optional[Text] = None,
                            base_image: Optional[Text] = None,
                            skaffold_cmd: Optional[Text] = None) -> Text:
    return builder.ContainerBuilder(
        target_image=target_image,
        base_image=base_image,
        skaffold_cmd=skaffold_cmd).build()

  def _get_pipeline_args(self, pipeline_name: Text,
                         arg_name: Text) -> Optional[Text]:
    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name)

    # Check if pipeline exists.
    self._check_pipeline_existence(pipeline_name)

    # TODO(b/157599419): Consider deprecating PipelineArgs.
    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path, _PIPELINE_ARG_FILE)
    # Get pipeline_id/experiment_id from pipeline_args.json
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)
    return pipeline_args.get(arg_name)

  def _get_pipeline_id(self, pipeline_name: Text) -> Text:
    pipeline_id = self._get_pipeline_args(pipeline_name, labels.PIPELINE_ID)
    if pipeline_id is None:
      raise ValueError(
          'Cannot find pipeline id for pipeline {}.'.format(pipeline_name))
    return pipeline_id

  def _print_runs(self, runs: List[Dict[Text, Any]]) -> None:
    """Prints runs in a tabular format with headers mentioned below."""
    headers = ('pipeline_name', 'job_name', 'status', 'created_at', 'link')
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    project_id = self.flags_dict[kubeflow_labels.GCP_PROJECT_ID_ENV]

    data = []
    for run in runs:
      data.append([
          pipeline_name,
          _get_job_name(run),
          run.get('state'),
          run.get('createTime'),
          _get_job_link(job_name=_get_job_name(run), project_id=project_id),
      ])

    click.echo(tabulate(data, headers=headers, tablefmt='grid'))
