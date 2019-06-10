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
"""Handler for Kubeflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click

from tfx.tools.cli.handler import base_handler


class KubeflowHandler(base_handler.BaseHandler):
  """Helper methods for Kubeflow Handler."""

  # TODO(b/132286477): Update comments after updating methods.

  def create_pipeline(self) -> None:
    """Creates pipeline in Kubeflow."""
    click.echo('Creating pipeline in Kubeflow')

  def update_pipeline(self) -> None:
    """Updates pipeline in Kubeflow."""
    click.echo('Updating pipeline in Kubeflow')

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    click.echo('List of pipelines in Kubeflow')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Kubeflow."""
    click.echo('Deleting pipeline in Kubeflow')

  def run_pipeline(self) -> None:
    """Run pipeline in Kubeflow."""
    click.echo('Triggering pipeline in Kubeflow')
