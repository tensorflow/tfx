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
"""Handler for Airflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click

from tfx.tools.cli.handler import base_handler


class AirflowHandler(base_handler.BaseHandler):
  """Helper methods for Airflow Handler."""

  # TODO(b/132286477): Update comments after updating methods.
  def create_pipeline(self) -> None:
    """Creates pipeline in Airflow."""
    click.echo('Creating pipeline in Airflow')

  def update_pipeline(self) -> None:
    """Updates pipeline in Airflow."""
    click.echo('Updating pipeline in Airflow')

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    click.echo('List of pipelines in Airflow')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Airflow."""
    click.echo('Deleting pipeline in Airflow')

  def run_pipeline(self) -> None:
    """Trigger DAG in Airflow."""
    click.echo('Triggering pipeline in Airflow')
