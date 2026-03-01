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
"""Commands for run group."""

from typing import Iterable, Dict

import click

from tfx.tools.cli import labels
from tfx.tools.cli.cli_context import Context
from tfx.tools.cli.cli_context import pass_context
from tfx.tools.cli.handler import handler_factory


@click.group('run')
def run_group() -> None:
  pass


@run_group.command('create', help='Create a new run for a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_name',
    '--pipeline-name',
    required=True,
    type=str,
    help='Name of the pipeline')
@click.option(
    '--endpoint',
    default='',
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default='',
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
@click.option(
    '--project',
    default='',
    type=str,
    help='GCP project ID that will be used to invoke Vertex Pipelines.'
)
@click.option(
    '--region',
    default='',
    type=str,
    help='GCP region that will be used to invoke Vertex Pipelines.'
)
@click.option(
    '--runtime_parameter',
    '--runtime-parameter',
    default=[],
    type=str,
    multiple=True,
    help='Runtime parameter for the next pipeline run.'
    ' Format: <parameter_name>=<parameter_value>'
)
def create_run(ctx: Context, engine: str, pipeline_name: str, endpoint: str,
               iap_client_id: str, namespace: str, project: str,
               region: str, runtime_parameter: Iterable[str]) -> None:
  """Command definition to create a pipeline run."""
  click.echo('Creating a run for pipeline: ' + pipeline_name)
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  ctx.flags_dict[labels.GCP_PROJECT_ID] = project
  ctx.flags_dict[labels.GCP_REGION] = region
  ctx.flags_dict[labels.RUNTIME_PARAMETER] = _parse_runtime_parameters(
      runtime_parameter)

  handler = handler_factory.create_handler(ctx.flags_dict)
  if (ctx.flags_dict[labels.ENGINE_FLAG]
      not in (labels.KUBEFLOW_ENGINE, labels.AIRFLOW_ENGINE,
              labels.VERTEX_ENGINE)) and runtime_parameter:
    raise NotImplementedError(
        'Currently runtime parameter is only supported in kubeflow, vertex, '
        'and airflow.')
  handler.create_run()


def _parse_runtime_parameters(
    runtime_parameters: Iterable[str]) -> Dict[str, str]:
  """Turns runtime parameter into dictionary."""
  result = {}
  for name_value_pair in runtime_parameters:
    if '=' not in name_value_pair:
      raise ValueError('Runtime parameter should be <name>=<value> format.')
    name, value = name_value_pair.split('=', maxsplit=1)
    result[name] = value
  return result


@run_group.command('terminate', help='Stop a run')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--run_id',
    '--run-id',
    required=True,
    type=str,
    help='Unique ID for the run.)')
@click.option(
    '--endpoint',
    default='',
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default='',
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def terminate_run(ctx: Context, engine: str, run_id: str, endpoint: str,
                  iap_client_id: str, namespace: str) -> None:
  """Command definition to stop a run."""
  click.echo('Terminating run.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.RUN_ID] = run_id
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).terminate_run()


@run_group.command('list', help='List all the runs of a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_name',
    '--pipeline-name',
    required=True,
    type=str,
    help='Name of the pipeline')
@click.option(
    '--endpoint',
    default='',
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default='',
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def list_runs(ctx: Context, engine: str, pipeline_name: str, endpoint: str,
              iap_client_id: str, namespace: str) -> None:
  """Command definition to list all runs of a pipeline."""
  click.echo('Listing all runs of pipeline: ' + pipeline_name)
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).list_runs()


@run_group.command('status', help='Get the status of a run.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_name',
    '--pipeline-name',
    required=True,
    type=str,
    help='Name of the pipeline')
@click.option(
    '--run_id',
    '--run-id',
    required=True,
    type=str,
    help='Unique ID for the run.')
@click.option(
    '--endpoint',
    default='',
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default='',
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
@click.option(
    '--project',
    default='',
    type=str,
    help='GCP project ID that will be used to invoke Vertex Pipelines.'
)
@click.option(
    '--region',
    default='',
    type=str,
    help='GCP region that will be used to invoke Vertex Pipelines.'
)
def get_run(ctx: Context, engine: str, pipeline_name: str, run_id: str,
            endpoint: str, iap_client_id: str, namespace: str, project: str,
            region: str) -> None:
  """Command definition to stop a run."""
  click.echo('Retrieving run status.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.RUN_ID] = run_id
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  ctx.flags_dict[labels.GCP_PROJECT_ID] = project
  ctx.flags_dict[labels.GCP_REGION] = region
  handler_factory.create_handler(ctx.flags_dict).get_run()


@run_group.command('delete', help='Delete a run')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--run_id',
    '--run-id',
    required=True,
    type=str,
    help='Unique ID for the run.')
@click.option(
    '--endpoint',
    default='',
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default='',
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def delete_run(ctx: Context, engine: str, run_id: str, endpoint: str,
               iap_client_id: str, namespace: str) -> None:
  """Command definition to delete a run."""
  click.echo('Deleting run.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.RUN_ID] = run_id
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).delete_run()
