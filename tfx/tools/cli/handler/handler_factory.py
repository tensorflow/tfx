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
"""Helper functions to choose engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import sys
from typing import Any, Dict, Text

import click

from tfx.tools.cli import labels
from tfx.tools.cli import pip_utils
from tfx.tools.cli.handler import base_handler


def detect_handler(flags_dict: Dict[Text, Any]) -> base_handler.BaseHandler:
  """Detect handler from the environment.

  Details:
    When the engine flag is set to 'auto', this method first finds all the
    packages in the local environment. The environment is first checked
    for multiple orchestrators and if true the user must rerun the command with
    required engine. If only one orchestrator is present, the engine is set to
    that.

  Args:
    flags_dict: A dictionary containing the flags of a command.

  Returns:
    Corrosponding Handler object.
  """
  packages_list = pip_utils.get_package_names()
  if (labels.AIRFLOW_PACKAGE_NAME in packages_list) and (
      labels.KUBEFLOW_PACKAGE_NAME in packages_list):
    sys.exit('Multiple orchestrators found. Choose one using --engine flag.')
  if labels.AIRFLOW_PACKAGE_NAME in packages_list:
    click.echo('Detected Airflow.')
    click.echo(
        'Use --engine flag if you intend to use a different orchestrator.')
    flags_dict[labels.ENGINE_FLAG] = 'airflow'
    from tfx.tools.cli.handler import airflow_handler  # pylint: disable=g-import-not-at-top
    return airflow_handler.AirflowHandler(flags_dict)
  elif labels.KUBEFLOW_PACKAGE_NAME in packages_list:
    click.echo('Detected Kubeflow.')
    click.echo(
        'Use --engine flag if you intend to use a different orchestrator.')
    flags_dict[labels.ENGINE_FLAG] = 'kubeflow'
    from tfx.tools.cli.handler import kubeflow_handler  # pylint: disable=g-import-not-at-top
    return kubeflow_handler.KubeflowHandler(flags_dict)
  else:
    click.echo('Detected Local.')
    click.echo(
        'Use --engine flag if you intend to use a different orchestrator.')
    flags_dict[labels.ENGINE_FLAG] = 'local'
    from tfx.tools.cli.handler import local_handler  # pylint: disable=g-import-not-at-top
    return local_handler.LocalHandler(flags_dict)


def create_handler(flags_dict: Dict[Text, Any]) -> base_handler.BaseHandler:
  """Retrieve handler from the environment using the --engine flag.

  Args:
    flags_dict: A dictionary containing the flags of a command.

  Raises:
    RuntimeError: When engine is not supported by TFX.

  Returns:
    Corresponding Handler object.
  """
  engine = flags_dict[labels.ENGINE_FLAG]
  packages_list = str(subprocess.check_output(['pip', 'freeze', '--local']))
  if engine == 'airflow':
    if labels.AIRFLOW_PACKAGE_NAME not in packages_list:
      sys.exit('Airflow not found.')
    from tfx.tools.cli.handler import airflow_handler  # pylint: disable=g-import-not-at-top
    return airflow_handler.AirflowHandler(flags_dict)
  elif engine == 'kubeflow':
    if labels.KUBEFLOW_PACKAGE_NAME not in packages_list:
      sys.exit('Kubeflow not found.')
    from tfx.tools.cli.handler import kubeflow_handler  # pylint: disable=g-import-not-at-top
    return kubeflow_handler.KubeflowHandler(flags_dict)
  elif engine == 'beam':
    from tfx.tools.cli.handler import beam_handler  # pylint: disable=g-import-not-at-top
    return beam_handler.BeamHandler(flags_dict)
  elif engine == 'local':
    from tfx.tools.cli.handler import local_handler  # pylint: disable=g-import-not-at-top
    return local_handler.LocalHandler(flags_dict)
  elif engine == 'auto':
    return detect_handler(flags_dict)
  else:
    raise RuntimeError('Engine {} is not supported.'.format(engine))
