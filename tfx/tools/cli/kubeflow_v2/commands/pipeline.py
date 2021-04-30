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
"""Commands for pipeline group."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text

import click
from tfx.tools.cli import labels
from tfx.tools.cli.cli_context import Context
from tfx.tools.cli.cli_context import pass_context
from tfx.tools.cli.kubeflow_v2 import labels as kubeflow_labels

try:
  from tfx.tools.cli.kubeflow_v2.handler import kubeflow_v2_handler  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


@click.group('pipeline')
def pipeline_group() -> None:
  pass


# TODO(b/149347293): Unify the CLI flags for different engines.
@pipeline_group.command('create', help='Create a pipeline')
@pass_context
@click.option(
    '--pipeline_path',
    '--pipeline-path',
    required=True,
    type=str,
    help='Path to Python DSL.')
@click.option(
    '--build_base_image',
    '--build-base-image',
    default=None,
    type=str,
    help='Container image path to be used as the base image. If not specified, '
    'target image will be build based on the released TFX image.')
@click.option(
    '--build_image',
    '--build-image',
    is_flag=True,
    default=False,
    help='Build a container image for the pipeline using Dockerfile in the '
    'current directory. If Dockerfile does not exist, a default Dockerfile '
    'will be generated using --build-base-image.')
def create_pipeline(ctx: Context, pipeline_path: Text, build_base_image: Text,
                    build_image: bool) -> None:
  """Command definition to create a pipeline."""
  click.echo('Creating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = kubeflow_labels.KUBEFLOW_V2_ENGINE
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.BUILD_IMAGE] = build_image
  ctx.flags_dict[labels.BASE_IMAGE] = build_base_image
  kubeflow_v2_handler.KubeflowV2Handler(ctx.flags_dict).create_pipeline()


# TODO(b/149347293): Unify the CLI flags for different engines.
@pipeline_group.command('update', help='Update an existing pipeline.')
@pass_context
@click.option(
    '--pipeline_path',
    '--pipeline-path',
    required=True,
    type=str,
    help='Path to Python DSL file')
@click.option(
    '--build_image',
    '--build-image',
    is_flag=True,
    default=False,
    help='Build a container image for the pipeline using Dockerfile in the '
    'current directory. If Dockerfile does not exist, a default Dockerfile '
    'will be generated using --build-base-image.')
def update_pipeline(ctx: Context, pipeline_path: Text,
                    build_image: bool) -> None:
  """Command definition to update a pipeline."""
  click.echo('Updating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = kubeflow_labels.KUBEFLOW_V2_ENGINE
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  ctx.flags_dict[labels.BUILD_IMAGE] = build_image
  kubeflow_v2_handler.KubeflowV2Handler(ctx.flags_dict).update_pipeline()


@pipeline_group.command('list', help='List all the pipelines')
@pass_context
def list_pipelines(ctx: Context) -> None:
  """Command definition to list pipelines."""
  click.echo('Listing all pipelines')
  ctx.flags_dict[labels.ENGINE_FLAG] = kubeflow_labels.KUBEFLOW_V2_ENGINE
  kubeflow_v2_handler.KubeflowV2Handler(ctx.flags_dict).list_pipelines()


@pipeline_group.command('delete', help='Delete a pipeline')
@pass_context
@click.option(
    '--pipeline_name',
    '--pipeline-name',
    required=True,
    type=str,
    help='Name of the pipeline')
def delete_pipeline(ctx: Context, pipeline_name: Text) -> None:
  """Command definition to delete a pipeline."""
  click.echo('Deleting pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = kubeflow_labels.KUBEFLOW_V2_ENGINE
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  kubeflow_v2_handler.KubeflowV2Handler(ctx.flags_dict).delete_pipeline()


@pipeline_group.command('compile', help='Compile a pipeline.')
@pass_context
@click.option(
    '--pipeline_path',
    '--pipeline-path',
    required=True,
    type=str,
    help='Path to Python DSL file.')
def compile_pipeline(ctx: Context, pipeline_path: Text) -> None:
  """Command definition to compile a pipeline."""
  click.echo('Compiling pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = kubeflow_labels.KUBEFLOW_V2_ENGINE
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path

  kubeflow_v2_handler.KubeflowV2Handler(ctx.flags_dict).compile_pipeline()
