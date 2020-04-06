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
"""Commands for run group."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

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
def create_run(ctx: Context, engine: Text, pipeline_name: Text, endpoint: Text,
               iap_client_id: Text, namespace: Text) -> None:
  """Command definition to create a pipeline run."""
  click.echo('Creating a run for pipeline: ' + pipeline_name)
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).create_run()


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
def terminate_run(ctx: Context, engine: Text, run_id: Text, endpoint: Text,
                  iap_client_id: Text, namespace: Text) -> None:
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
def list_runs(ctx: Context, engine: Text, pipeline_name: Text, endpoint: Text,
              iap_client_id: Text, namespace: Text) -> None:
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
def get_run(ctx: Context, engine: Text, pipeline_name: Text, run_id: Text,
            endpoint: Text, iap_client_id: Text, namespace: Text) -> None:
  """Command definition to stop a run."""
  click.echo('Retrieving run status.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.RUN_ID] = run_id
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
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
def delete_run(ctx: Context, engine: Text, run_id: Text, endpoint: Text,
               iap_client_id: Text, namespace: Text) -> None:
  """Command definition to delete a run."""
  click.echo('Deleting run.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.RUN_ID] = run_id
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).delete_run()
