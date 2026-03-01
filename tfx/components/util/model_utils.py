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
"""Common functionalities used in several components."""

from tfx import types


def is_model_blessed(model_blessing: types.Artifact) -> bool:
  """Returns whether model is blessed by upstream ModelValidator.

  Args:
    model_blessing: model blessing artifact from model_validator.

  Returns:
    True if the model is blessed by validator.
  """
  return model_blessing.get_int_custom_property('blessed') == 1


# TODO(jjong): Current pusher doesn't check whether the infra validated
# environment is identical to the pushing environment, and it is a user's
# responsibility to ensure the alignment. We need to provide a better mechanism.
def is_infra_validated(infra_blessing: types.Artifact) -> bool:
  """Returns whether model is infra blessed by upstream InfraValidator.

  Args:
    infra_blessing: A `InfraBlessing` artifact from infra validator.

  Returns:
    Whether model is infra validated or not.
  """
  return infra_blessing.get_int_custom_property('blessed') == 1
