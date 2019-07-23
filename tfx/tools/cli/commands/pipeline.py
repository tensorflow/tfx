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

import click
from typing import Text

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
    '--path',
    'pipeline_path',
    required=True,
    type=str,
    help='Path to Python DSL.')
def create_pipeline(ctx: Context, engine: Text, pipeline_path: Text) -> None:
  """Command definition to create a pipeline."""
  click.echo('Creating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  handler_factory.create_handler(ctx.flags_dict).create_pipeline()


@pipeline_group.command('update', help='Update an existing pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--path',
    'pipeline_path',
    required=True,
    type=str,
    help='Path to Python DSL file')
def update_pipeline(ctx: Context, engine: Text, pipeline_path: Text) -> None:
  """Command definition to update a pipeline."""
  click.echo('Updating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  handler_factory.create_handler(ctx.flags_dict).update_pipeline()


@pipeline_group.command('delete', help='Delete a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--name',
    'pipeline_name',
    required=True,
    type=str,
    help='Name of the pipeline')
def delete_pipeline(ctx: Context, pipeline_name: Text, engine: Text) -> None:
  """Command definition to delete a pipeline."""
  click.echo('Deleting pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  handler_factory.create_handler(ctx.flags_dict).delete_pipeline()


@pipeline_group.command('list', help='List all the pipelines')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='orchestrator for pipelines')
def list_pipelines(ctx: Context, engine: Text) -> None:
  """Command definition to list pipelines."""
  click.echo('Listing all pipelines')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  handler_factory.create_handler(ctx.flags_dict).list_pipelines()


# TODO(b/132286477) Add compile_pipeline method to all handlers.
@pipeline_group.command('compile', help='Compile a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--path',
    'pipeline_path',
    required=True,
    type=str,
    help='Path to Python DSL.')
def compile_pipeline(ctx: Context, engine: Text, pipeline_path: Text) -> None:
  """Command definition to create a pipeline."""
  click.echo('Compiling pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_DSL_PATH] = pipeline_path
  handler_factory.create_handler(ctx.flags_dict).compile_pipeline()
