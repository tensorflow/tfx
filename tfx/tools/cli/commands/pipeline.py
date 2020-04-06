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
"""Commands for pipeline group."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import click

from tfx.tools.cli import labels
from tfx.tools.cli.cli_context import Context
from tfx.tools.cli.cli_context import pass_context
from tfx.tools.cli.handler import handler_factory


@click.group('pipeline')
def pipeline_group() -> None:
  pass


# TODO(b/132286477): Add support for requirements file.
@pipeline_group.command('create', help='Create a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_path',
    '--pipeline-path',
    required=True,
    type=str,
    help='Path to Python DSL.')
@click.option(
    '--package_path',
    '--package-path',
    default=None,
    type=str,
    help='Path to the pipeline output workflow file. When unset, it will try to find the workflow file, "<pipeline_name>.tar.gz" in the current directory.'
)
@click.option(
    '--build_target_image',
    '--build-target-image',
    default=None,
    type=str,
    help='Target container image path. The target image will be built by this '
    'command to include local python codes to the TFX default image. By default, '
    'it uses docker daemon to build an image which will install the local '
    'python setup file onto TFX default image. You can place a setup.py file '
    'to control the python code to install the dependent packages. You can also '
    'customize the Skaffold building options by placing a build.yaml in the '
    'local directory. In addition, you can place a Dockerfile file to customize'
    'the docker building script.'
)
@click.option(
    '--build_base_image',
    '--build-base-image',
    default=None,
    type=str,
    help='Container image path to be used as the base image. If not specified, '
    'target image will be build based on the released TFX image.'
)
@click.option(
    '--skaffold_cmd',
    '--skaffold-cmd',
    default=None,
    type=str,
    help='Skaffold program command.')
@click.option(
    '--endpoint',
    default=None,
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default=None,
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def create_pipeline(ctx: Context, engine: Text, pipeline_path: Text,
                    package_path: Text, build_target_image: Text,
                    build_base_image: Text,
                    skaffold_cmd: Text, endpoint: Text, iap_client_id: Text,
                    namespace: Text) -> None:
  """Command definition to create a pipeline."""
  # TODO(b/142358865): Add support for container building for Airflow and Beam
  # runners when they support container executors.
  click.echo('Creating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.PIPELINE_PACKAGE_PATH] = package_path
  ctx.flags_dict[labels.TARGET_IMAGE] = build_target_image
  ctx.flags_dict[labels.BASE_IMAGE] = build_base_image
  ctx.flags_dict[labels.SKAFFOLD_CMD] = skaffold_cmd
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).create_pipeline()


@pipeline_group.command('update', help='Update an existing pipeline.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_path',
    '--pipeline-path',
    required=True,
    type=str,
    help='Path to Python DSL file')
@click.option(
    '--package_path',
    '--package-path',
    type=str,
    default=None,
    help='Path to the pipeline output workflow file. When unset, it will try to find the workflow file, "<pipeline_name>.tar.gz" in the current directory.'
)
@click.option(
    '--skaffold_cmd',
    '--skaffold-cmd',
    default=None,
    type=str,
    help='Skaffold program command.')
@click.option(
    '--endpoint',
    default=None,
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default=None,
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def update_pipeline(ctx: Context, engine: Text, pipeline_path: Text,
                    package_path: Text, skaffold_cmd: Text,
                    endpoint: Text, iap_client_id: Text,
                    namespace: Text) -> None:
  """Command definition to update a pipeline."""
  click.echo('Updating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.PIPELINE_PACKAGE_PATH] = package_path
  ctx.flags_dict[labels.SKAFFOLD_CMD] = skaffold_cmd
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).update_pipeline()


@pipeline_group.command('delete', help='Delete a pipeline')
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
    default=None,
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default=None,
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def delete_pipeline(ctx: Context, engine: Text, pipeline_name: Text,
                    endpoint: Text, iap_client_id: Text,
                    namespace: Text) -> None:
  """Command definition to delete a pipeline."""
  click.echo('Deleting pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).delete_pipeline()


@pipeline_group.command('list', help='List all the pipelines')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='orchestrator for pipelines')
@click.option(
    '--endpoint',
    default=None,
    type=str,
    help='Endpoint of the KFP API service to connect.')
@click.option(
    '--iap_client_id',
    '--iap-client-id',
    default=None,
    type=str,
    help='Client ID for IAP protected endpoint.')
@click.option(
    '-n',
    '--namespace',
    default='kubeflow',
    type=str,
    help='Kubernetes namespace to connect to the KFP API.')
def list_pipelines(ctx: Context, engine: Text, endpoint: Text,
                   iap_client_id: Text, namespace: Text) -> None:
  """Command definition to list pipelines."""
  click.echo('Listing all pipelines')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  handler_factory.create_handler(ctx.flags_dict).list_pipelines()


@pipeline_group.command('compile', help='Compile a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_path',
    '--pipeline-path',
    required=True,
    type=str,
    help='Path to Python DSL.')
@click.option(
    '--package_path',
    '--package-path',
    default=None,
    type=str,
    help='Path to the pipeline output workflow file. When unset, it will try to find the workflow file, "<pipeline_name>.tar.gz" in the current directory.'
)
def compile_pipeline(ctx: Context, engine: Text, pipeline_path: Text,
                     package_path: Text) -> None:
  """Command definition to compile a pipeline."""
  click.echo('Compiling pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.PIPELINE_PACKAGE_PATH] = package_path
  handler_factory.create_handler(ctx.flags_dict).compile_pipeline()


@pipeline_group.command('schema', help='Obtain latest database schema.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--pipeline_name',
    '--pipeline-name',
    required=True,
    type=str,
    help='Name of the pipeline')
def get_schema(ctx: Context, engine: Text, pipeline_name: Text) -> None:
  """Command definition to infer latest schema."""
  click.echo('Getting latest schema.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  handler_factory.create_handler(ctx.flags_dict).get_schema()
