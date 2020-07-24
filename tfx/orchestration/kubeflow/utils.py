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
"""Common utility for Kubeflow-based orchestrator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from typing import Text
# utils.py should not be used in container_entrypoint.py because of its
# dependency on KFP.
from kfp import dsl

from tfx.orchestration import data_types
from tfx.utils import json_utils


def replace_placeholder(serialized_component: Text) -> Text:
  """Replaces the RuntimeParameter placeholders with kfp.dsl.PipelineParam."""
  placeholders = re.findall(data_types.RUNTIME_PARAMETER_PATTERN,
                            serialized_component)

  for placeholder in placeholders:
    # We need to keep the level of escaping of original RuntimeParameter
    # placeholder. This can be done by probing the pair of quotes around
    # literal 'RuntimeParameter'.
    placeholder = fix_brackets(placeholder)
    cleaned_placeholder = placeholder.replace('\\', '')  # Clean escapes.
    parameter = json_utils.loads(cleaned_placeholder)
    dsl_parameter_str = str(dsl.PipelineParam(name=parameter.name))

    serialized_component = serialized_component.replace(placeholder,
                                                        dsl_parameter_str)

  return serialized_component


def fix_brackets(placeholder: Text) -> Text:
  """Fix the imbalanced brackets in placeholder.

  When ptype is not null, regex matching might grab a placeholder with }
  missing. This function fix the missing bracket.

  Args:
    placeholder: string placeholder of RuntimeParameter

  Returns:
    Placeholder with re-balanced brackets.

  Raises:
    RuntimeError: if left brackets are less than right brackets.
  """
  lcount = placeholder.count('{')
  rcount = placeholder.count('}')
  if lcount < rcount:
    raise RuntimeError(
        'Unexpected redundant left brackets found in {}'.format(placeholder))
  else:
    patch = ''.join(['}'] * (lcount - rcount))
    return placeholder + patch
