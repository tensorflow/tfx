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

import sys
from typing import Optional

import click

from tfx.tools.cli import labels
from tfx.tools.cli.cli_context import Context
from tfx.tools.cli.cli_context import pass_context
from tfx.tools.cli.handler import handler_factory


def _check_deprecated_image_build_flags(build_target_image=None,
                                        skaffold_cmd=None,
                                        pipeline_package_path=None):
  """Checks and exits if deprecated flags were used."""
  if build_target_image is not None:
    sys.exit(
        '[Error] --build-target-image flag was DELETED. You should specify '
        'the build target image at the `KubeflowDagRunnerConfig` class '
        'instead, and use --build-image flag without argument to build a '
        'container image when creating or updating a pipeline.')

  if skaffold_cmd is not None:
    sys.exit(
        '[Error] --skaffold-cmd flag was DELETED. TFX doesn\'t use skaffold '
        'any more. You can delete --skaffold-cmd flag and the auto-genrated '
        'build.yaml file. You must specify --build-image to trigger an '
        'image build when creating or updating a pipeline.')

  if pipeline_package_path is not None:
    sys.exit(
        '[Error] --pipeline-package-path flag was DELETED. You can specify '
        'the package location as `output_filename` and `output_dir` when '
        'creating a `KubeflowDagRunner` instance. CLI will read the pacakge '
        'path specified there.')


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
    help='[DEPRECATED] Package path specified in a KubeflowDagRunner instace '
    'will be used.')
@click.option(
    '--build_target_image',
    '--build-target-image',
    default=None,
    type=str,
    help='[DEPRECATED] Please specify target image to the '
    'KubeflowDagRunnerConfig class directly. `KUBEFLOW_TFX_IMAGE` environment '
    'variable is not used any more.')
@click.option(
    '--build_base_image',
    '--build-base-image',
    default=None,
    type=str,
    help='Container image path to be used as the base image. If not specified, '
    'official TFX image with the same version will be used. You need to '
    'specify --build-image flag to trigger an image build.')
@click.option(
    '--skaffold_cmd',
    '--skaffold-cmd',
    default=None,
    type=str,
    help='[DEPRECATED] Skaffold is not used any more. Do not use this flag.')
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
@click.option(
    '--build_image',
    '--build-image',
    is_flag=True,
    default=False,
    help='Build a container image for the pipeline using Dockerfile in the '
    'current directory. If Dockerfile does not exist, a default Dockerfile '
    'will be generated using --build-base-image.')
def create_pipeline(ctx: Context, engine: str, pipeline_path: str,
                    package_path: Optional[str],
                    build_target_image: Optional[str],
                    build_base_image: Optional[str],
                    skaffold_cmd: Optional[str], endpoint: Optional[str],
                    iap_client_id: Optional[str], namespace: str,
                    build_image: bool) -> None:
  """Command definition to create a pipeline."""
  # TODO(b/179847638): Delete checks for deprecated flags.
  _check_deprecated_image_build_flags(build_target_image, skaffold_cmd,
                                      package_path)

  if build_base_image is not None and not build_image:
    sys.exit('--build-base-image used without --build-image. You have to use '
             '--build-image flag to build a container image for the pipeline.')

  # TODO(b/142358865): Add support for container building for Airflow and Beam
  # runners when they support container executors.
  click.echo('Creating pipeline')

  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.BASE_IMAGE] = build_base_image
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  ctx.flags_dict[labels.BUILD_IMAGE] = build_image
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
    help='[DEPRECATED] Package path specified in a KubeflowDagRunner instace '
    'will be used.')
@click.option(
    '--skaffold_cmd',
    '--skaffold-cmd',
    default=None,
    type=str,
    help='[DEPRECATED] Skaffold is not used any more. Do not use this flag.')
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
@click.option(
    '--build_image',
    '--build-image',
    is_flag=True,
    default=False,
    help='Build a container image for the pipeline using Dockerfile in the '
    'current directory.')
def update_pipeline(ctx: Context, engine: str, pipeline_path: str,
                    package_path: Optional[str], skaffold_cmd: Optional[str],
                    endpoint: Optional[str], iap_client_id: Optional[str],
                    namespace: str, build_image: bool) -> None:
  """Command definition to update a pipeline."""
  # TODO(b/179847638): Delete checks for deprecated flags.
  _check_deprecated_image_build_flags(None, skaffold_cmd, package_path)

  click.echo('Updating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.ENDPOINT] = endpoint
  ctx.flags_dict[labels.IAP_CLIENT_ID] = iap_client_id
  ctx.flags_dict[labels.NAMESPACE] = namespace
  ctx.flags_dict[labels.BUILD_IMAGE] = build_image
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
def delete_pipeline(ctx: Context, engine: str, pipeline_name: str,
                    endpoint: str, iap_client_id: str, namespace: str) -> None:
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
def list_pipelines(ctx: Context, engine: str, endpoint: str, iap_client_id: str,
                   namespace: str) -> None:
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
    help='[DEPRECATED] Package path specified in a KubeflowDagRunner instace '
    'will be used.')
def compile_pipeline(ctx: Context, engine: str, pipeline_path: str,
                     package_path: str) -> None:
  """Command definition to compile a pipeline."""
  # TODO(b/179847638): Delete checks for deprecated flags.
  _check_deprecated_image_build_flags(pipeline_package_path=package_path)

  click.echo('Compiling pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
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
def get_schema(ctx: Context, engine: str, pipeline_name: str) -> None:
  """Command definition to infer latest schema."""
  click.echo('Getting latest schema.')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  handler_factory.create_handler(ctx.flags_dict).get_schema()
