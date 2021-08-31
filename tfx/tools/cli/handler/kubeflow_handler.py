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

import functools

import os
import sys
import time
from typing import Any, Dict, Optional

import click
import kfp

from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.tools.cli import labels
from tfx.tools.cli.container_builder import builder
from tfx.tools.cli.handler import base_handler
from tfx.tools.cli.handler import kubeflow_dag_runner_patcher


def create_container_image(image: str, base_image: Optional[str]) -> str:
  if image == kubeflow_dag_runner.DEFAULT_KUBEFLOW_TFX_IMAGE:
    sys.exit('Default image for KubeflowDagRunner given and used with '
             '--build-image flag. If you want to use your custom image, please '
             'specify the image name that will be built at the '
             'KubeflowDagRunnerConfig. Otherwise, do not use --build-image '
             'flag.')
  built_image = builder.build(target_image=image, base_image=base_image)
  click.echo(f'New container image "{built_image}" was built.')
  return built_image


# TODO(b/132286477): Change generated api methods to client methods after SDK is
# updated.
class KubeflowHandler(base_handler.BaseHandler):
  """Helper methods for Kubeflow Handler."""

  def __init__(self, flags_dict: Dict[str, Any]):
    """Initialize Kubeflow handler.

    Args:
      flags_dict: A dictionary with flags provided in a command.
    """
    super().__init__(flags_dict)
    # TODO(b/132286477): Change to setup config instead of flags if needed.
    if labels.NAMESPACE in self.flags_dict:
      self._client = kfp.Client(
          host=self.flags_dict[labels.ENDPOINT],
          client_id=self.flags_dict[labels.IAP_CLIENT_ID],
          namespace=self.flags_dict[labels.NAMESPACE])
    else:
      self._client = None

  def create_pipeline(self, update: bool = False) -> None:
    """Creates or updates a pipeline in Kubeflow.

    Args:
      update: set as true to update pipeline.
    """

    if self.flags_dict.get(labels.BUILD_IMAGE):
      build_image_fn = functools.partial(
          create_container_image,
          base_image=self.flags_dict.get(labels.BASE_IMAGE))
    else:
      build_image_fn = None
      if os.path.exists('build.yaml'):
        click.echo(
            '[Warning] TFX doesn\'t depend on skaffold anymore and you can '
            'delete the auto-genrated build.yaml file. TFX will NOT build a '
            'container even if build.yaml file exists. Use --build-image flag '
            'to trigger an image build when creating or updating a pipeline.',
            err=True)
    patcher = kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher(
        call_real_run=True,
        use_temporary_output_file=True,
        build_image_fn=build_image_fn)
    context = self.execute_dsl(patcher)
    pipeline_name = context[patcher.PIPELINE_NAME]
    pipeline_package_path = context[patcher.OUTPUT_FILE_PATH]

    self._save_pipeline(pipeline_name, pipeline_package_path, update=update)

    if context[patcher.USE_TEMPORARY_OUTPUT_FILE]:
      os.remove(pipeline_package_path)

    if update:
      click.echo('Pipeline "{}" updated successfully.'.format(pipeline_name))
    else:
      click.echo('Pipeline "{}" created successfully.'.format(pipeline_name))

  def update_pipeline(self) -> None:
    """Updates pipeline in Kubeflow."""
    self.create_pipeline(update=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    response = self._client.list_pipelines(page_size=100)

    if response and response.pipelines:
      click.echo(response.pipelines)
    else:
      click.echo('No pipelines to display.')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""

    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    # Check if pipeline exists on server.
    pipeline_id = self._get_pipeline_id(pipeline_name, check=True)

    # Delete pipeline for kfp server.
    self._client.delete_pipeline(pipeline_id)

    # Delete experiment from server.
    experiment_id = self._get_experiment_id(
        self._get_experiment_name(pipeline_name))
    self._client._experiment_api.delete_experiment(experiment_id)  # pylint: disable=protected-access

    click.echo('Pipeline ' + pipeline_name + ' deleted successfully.')

  def compile_pipeline(self) -> None:
    """Compiles pipeline in Kubeflow.

    Returns:
      pipeline_args: python dictionary with pipeline details extracted from DSL.
    """
    patcher = kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher(
        call_real_run=True,
        use_temporary_output_file=False)
    context = self.execute_dsl(patcher)

    click.echo('Pipeline compiled successfully.')
    click.echo(f'Pipeline package path: {context[patcher.OUTPUT_FILE_PATH]}')

  def create_run(self) -> None:
    """Runs a pipeline in Kubeflow."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    experiment_name = self._get_experiment_name(pipeline_name)

    pipeline_id = self._get_pipeline_id(pipeline_name)
    pipeline_version_id = self._get_pipeline_version_id(pipeline_id)
    experiment_id = self._get_experiment_id(experiment_name)

    runtime_parameters_dict = self.flags_dict[labels.RUNTIME_PARAMETER]

    # Run pipeline.
    run = self._client.run_pipeline(
        experiment_id=experiment_id,
        job_name=self._get_run_job_name(pipeline_name),
        params=runtime_parameters_dict,
        version_id=pipeline_version_id)

    click.echo('Run created for pipeline: ' + pipeline_name)
    self._print_runs(pipeline_name, [run])

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
    self._get_pipeline_id(pipeline_name, check=True)

    # List runs.
    experiment_id = self._get_experiment_id(
        self._get_experiment_name(pipeline_name))
    response = self._client.list_runs(experiment_id=experiment_id)

    if response and response.runs:
      self._print_runs(pipeline_name, response.runs)
    else:
      click.echo('No runs found.')

  def get_run(self) -> None:
    """Checks run status."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    run = self._client.get_run(self.flags_dict[labels.RUN_ID]).run
    self._print_runs(pipeline_name, [run])

  def get_schema(self):
    patcher = kubeflow_dag_runner_patcher.KubeflowDagRunnerPatcher(
        call_real_run=False)
    context = self.execute_dsl(patcher)

    self._read_schema_from_pipeline_root(context[patcher.PIPELINE_NAME],
                                         context[patcher.PIPELINE_ROOT])

  def _get_experiment_name(self, pipeline_name: str) -> str:
    return pipeline_name

  def _get_run_job_name(self, pipeline_name: str) -> str:
    return pipeline_name

  def _get_pipeline_id(self,
                       pipeline_name: str,
                       check: bool = True) -> Optional[None]:
    pipeline_id = self._client.get_pipeline_id(pipeline_name)
    if check and pipeline_id is None:
      sys.exit(f'Cannot find pipeline "{pipeline_name}". Did you provide '
               'correct endpoint and pipeline name?')
    return pipeline_id

  def _save_pipeline(self,
                     pipeline_name: str,
                     pipeline_package_path: str,
                     update: bool = False) -> None:
    """Creates/updates pipeline in the Kubeflow Pipelines cluster."""

    pipeline_id = self._get_pipeline_id(pipeline_name, check=update)
    if pipeline_id is not None and not update:
      sys.exit(
          f'Pipeline "{pipeline_name}" already exists. id="{pipeline_id}")')

    if update:
      # A timestamp will be appended for the uniqueness of `version_name`.
      version_name = '{}_{}'.format(pipeline_name,
                                    time.strftime('%Y%m%d%H%M%S'))
      self._client.upload_pipeline_version(
          pipeline_package_path=pipeline_package_path,
          pipeline_version_name=version_name,
          pipeline_id=pipeline_id)
    else:  # creating a new pipeline.
      upload_response = self._client.upload_pipeline(
          pipeline_package_path=pipeline_package_path,
          pipeline_name=pipeline_name)
      pipeline_id = upload_response.id

      # Create experiment with pipeline name as experiment name.
      self._client.create_experiment(pipeline_name)

    # Display the link to the pipeline detail page in KFP UI.
    click.echo('Please access the pipeline detail page at '
               '{prefix}/#/pipelines/details/{pipeline_id}'.format(
                   prefix=self._client._get_url_prefix(),  # pylint: disable=protected-access
                   pipeline_id=pipeline_id))

  def _get_pipeline_version_id(self, pipeline_id: str) -> Optional[str]:
    # Get the latest version.
    response = self._client.list_pipeline_versions(
        pipeline_id, page_size=1, sort_by='created_at desc')
    assert len(response.versions) == 1
    return response.versions[0].id

  def _get_experiment_id(self, pipeline_name: str) -> str:
    experiment = self._client.get_experiment(
        experiment_name=self._get_experiment_name(pipeline_name))
    return experiment.id

  def _print_runs(self, pipeline_name, runs):
    """Prints runs in a tabular format with headers mentioned below."""
    headers = ['pipeline_name', 'run_id', 'status', 'created_at', 'link']

    def _get_run_details(run_id):
      """Return the link to the run detail page."""
      return '{prefix}/#/runs/details/{run_id}'.format(
          prefix=self._client._get_url_prefix(), run_id=run_id)  # pylint: disable=protected-access

    data = [
        [  # pylint: disable=g-complex-comprehension
            pipeline_name, run.id, run.status,
            run.created_at.isoformat(),
            _get_run_details(run.id)
        ] for run in runs
    ]
    click.echo(self._format_table(headers, data))
