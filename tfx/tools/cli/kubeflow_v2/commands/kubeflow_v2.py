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
"""Entrypoint for Kubeflow V2 CLI commands."""

import click

from tfx.tools.cli.kubeflow_v2.commands import pipeline
from tfx.tools.cli.kubeflow_v2.commands import run


# TODO(b/172966765): Merge this click group with other CLI commands.
@click.group(
    'kubeflow_v2',
    help='[Experimental] V2 Kubeflow command line tool.')
def kubeflow_v2_group():
  click.echo('Kubeflow V2')


kubeflow_v2_group.add_command(pipeline.pipeline_group)
kubeflow_v2_group.add_command(run.run_group)
