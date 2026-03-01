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
"""Commands for copy_template."""

import click

from tfx.tools.cli import labels
from tfx.tools.cli.cli_context import Context
from tfx.tools.cli.cli_context import pass_context
from tfx.tools.cli.handler import template_handler


@click.group(
    'template',
    help='[Experimental] Helps creating a new TFX pipeline scaffold.')
def template_group() -> None:
  pass


@template_group.command('list', help='[Experimental] List available templates')
def list_templates() -> None:
  click.echo('Available templates:')
  for model in template_handler.list_template():
    click.echo('- {}'.format(model))


@template_group.command(
    'copy', help='[Experimental] Copy a template to destination directory')
@pass_context
@click.option(
    '--pipeline_name',
    '--pipeline-name',
    required=True,
    type=str,
    help='Name of the pipeline')
@click.option(
    '--destination_path',
    '--destination-path',
    required=True,
    type=str,
    help='Destination directory path to copy the pipeline template')
@click.option(
    '--model',
    required=True,
    type=str,
    help='Name of the template to copy. Currently, `taxi` is the only template provided.'
)
def copy(ctx: Context, pipeline_name: str, destination_path: str,
         model: str) -> None:
  """Command definition to copy template to specified directory."""
  click.echo('Copying {} pipeline template'.format(model))
  ctx.flags_dict[labels.PIPELINE_NAME] = pipeline_name
  ctx.flags_dict[labels.DESTINATION_PATH] = destination_path
  ctx.flags_dict[labels.MODEL] = model
  template_handler.copy_template(ctx.flags_dict)
