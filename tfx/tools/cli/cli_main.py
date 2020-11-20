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
"""Main script to invoke CLI."""

import click

from tfx.tools.cli.commands.pipeline import pipeline_group
from tfx.tools.cli.commands.run import run_group
from tfx.tools.cli.commands.template import template_group
from tfx.tools.cli.kubeflow_v2.commands import kubeflow_v2


@click.group('cli')
def cli_group():
  click.echo('CLI')


cli_group.add_command(pipeline_group)
cli_group.add_command(run_group)
cli_group.add_command(template_group)
cli_group.add_command(kubeflow_v2.kubeflow_v2_group)


if __name__ == '__main__':
  cli_group()
