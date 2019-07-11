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
import click

from typing import Dict, Text, Any
from tfx.tools.cli import labels
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
  packages_list = str(subprocess.check_output(['pip', 'freeze', '--local']))
  if labels.AIRFLOW_PACKAGE_NAME in packages_list and labels.KUBEFLOW_PACKAGE_NAME in packages_list:
    sys.exit('Multiple orchestrators found. Choose one using --engine flag.')
  if labels.AIRFLOW_PACKAGE_NAME in packages_list:
    click.echo('Detected Airflow.')
    flags_dict[labels.ENGINE_FLAG] = 'airflow'
    from tfx.tools.cli.handler import airflow_handler  # pylint: disable=g-import-not-at-top
    return airflow_handler.AirflowHandler(flags_dict)
  elif labels.KUBEFLOW_PACKAGE_NAME in packages_list:
    click.echo('Detected Kubeflow.')
    flags_dict[labels.ENGINE_FLAG] = 'kubeflow'
    from tfx.tools.cli.handler import kubeflow_handler  # pylint: disable=g-import-not-at-top
    return kubeflow_handler.KubeflowHandler(flags_dict)
  # TODO(b/132286477):Update to beam runner later.
  sys.exit('Orchestrator missing in the environment.')


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
  if engine == 'airflow':
    from tfx.tools.cli.handler import airflow_handler  # pylint: disable=g-import-not-at-top
    return airflow_handler.AirflowHandler(flags_dict)
  elif engine == 'kubeflow':
    from tfx.tools.cli.handler import kubeflow_handler  # pylint: disable=g-import-not-at-top
    return kubeflow_handler.KubeflowHandler(flags_dict)
  elif engine == 'auto':
    return detect_handler(flags_dict)
  else:
    raise RuntimeError('Engine {} is not supported.'.format(engine))
