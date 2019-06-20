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

from typing import Dict, Text, Any

from tfx.tools.cli import labels
from tfx.tools.cli.handler import airflow_handler
from tfx.tools.cli.handler import base_handler
from tfx.tools.cli.handler import kubeflow_handler


def detect_handler(flags_dict: Dict[Text, Any]) -> base_handler.BaseHandler:
  # TODO(b/132286477):Autodetect engine from environment
  """Detect handler from the environment.

  Args:
    flags_dict: A dictionary containing the flags of a command.

  Returns: Corrosponding Handler object.
  """
  raise NotImplementedError('Orchestrator '+flags_dict['engine']+
                            ' missing in the environment.')


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
    return airflow_handler.AirflowHandler(flags_dict)
  elif engine == 'kubeflow':
    return kubeflow_handler.KubeflowHandler(flags_dict)
  elif engine == 'auto':
    return detect_handler(flags_dict)
  else:
    raise RuntimeError('Engine {} is not supported.'.format(engine))
