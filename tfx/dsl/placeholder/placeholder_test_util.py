# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Utilities to deal with placeholders in unit tests."""

from typing import Any, Optional

from tfx.dsl.compiler import placeholder_utils
from tfx.dsl.placeholder import placeholder_base
from tfx.orchestration.portable import data_types


def resolve(
    placeholder: placeholder_base.Placeholder,
    resolution_context: Optional[placeholder_utils.ResolutionContext] = None,
) -> Any:
  """Resolves the given placeholder.

  Args:
    placeholder: The placeholder to resolve.
    resolution_context: Contextual information. This defaults to an empty
      context, which still allows _some_ placeholder expressions to be resolved.
      Note that resolution may fail if you do not provide a context with the
      appropriate information.

  Returns:
    The resolved value.
  """
  return placeholder_utils.resolve_placeholder_expression(
      placeholder.encode(),
      resolution_context
      or placeholder_utils.ResolutionContext(
          exec_info=data_types.ExecutionInfo()
      ),
  )


def maybe_resolve(
    maybe_placeholder: Any,
    resolution_context: Optional[placeholder_utils.ResolutionContext] = None,
) -> Any:
  """Calls resolve() on placeholders, passes through all other inputs."""
  if isinstance(maybe_placeholder, placeholder_base.Placeholder):
    return resolve(maybe_placeholder, resolution_context)
  return maybe_placeholder
