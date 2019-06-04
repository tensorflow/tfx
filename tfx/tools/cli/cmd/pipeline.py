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
from tfx.tools.cli.cmd import labels


class PipelineContext(object):

  def __init__(self):
    self.flags_dict = {}


pass_context = click.make_pass_decorator(PipelineContext, ensure=True)


@click.group('pipeline')
def pipeline_group() -> None:
  pass


@pipeline_group.command('create', help='Create a pipeline')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--path',
    'pipeline_path',
    required=True,
    type=str,
    help='Path to Python DSL file')
def create_pipeline(ctx: PipelineContext, engine: Text,
                    pipeline_path: Text) -> None:
  """Command definition create a pipeline."""
  click.echo('Creating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_PATH_FLAG] = pipeline_path


@pipeline_group.command('update', help='Update an existing pipeline.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--path',
    'pipeline_path',
    required=True,
    type=str,
    help='Path to Python DSL file')
def update_pipeline(ctx: PipelineContext, engine: Text,
                    pipeline_path: Text) -> None:
  """Command definition update a pipeline."""
  click.echo('Updating pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_PATH_FLAG] = pipeline_path


@pipeline_group.command('run', help='Create a new run for a pipeline.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--name',
    'pipeline_name',
    required=True,
    type=str,
    help='Name of the pipeline')
def run_pipeline(ctx: PipelineContext, pipeline_name: Text,
                 engine: Text) -> None:
  """Command definition run a pipeline."""
  click.echo('Triggering pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME_FLAG] = pipeline_name


@pipeline_group.command('delete', help='Delete a pipeline.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='Orchestrator for pipelines')
@click.option(
    '--name',
    'pipeline_name',
    required=True,
    type=str,
    help='Name of the pipeline')
def delete_pipeline(ctx: PipelineContext, pipeline_name: Text,
                    engine: Text) -> None:
  """Command definition delete a pipeline."""
  click.echo('Deleting pipeline')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
  ctx.flags_dict[labels.PIPELINE_NAME_FLAG] = pipeline_name


@pipeline_group.command('list', help='List all the pipelines.')
@pass_context
@click.option(
    '--engine', default='auto', type=str, help='orchestrator for pipelines')
def list_pipelines(ctx: PipelineContext, engine: Text) -> None:
  """Command definition list pipelines."""
  click.echo('Listing all pipelines')
  ctx.flags_dict[labels.ENGINE_FLAG] = engine
