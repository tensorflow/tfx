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
from kfp import dsl

from tfx.orchestration.experimental.runtime_parameter import runtime_string_parameter


def replace_placeholder(serialized_component: Text) -> Text:
  """Replaces the RuntimeParameter placeholders with kfp.dsl.PipelineParam."""
  placeholders = re.findall(runtime_string_parameter.PARAMETER_PATTERN,
                            serialized_component)

  for placeholder in placeholders:
    parameter = runtime_string_parameter.RuntimeStringParameter.parse(
        placeholder)
    dsl_parameter = dsl.PipelineParam(name=parameter.name)
    serialized_component = serialized_component.replace(placeholder,
                                                        str(dsl_parameter))

  return serialized_component
