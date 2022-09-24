# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Module for public facing, canned resolver functions."""

from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution.ops import ops


@resolver_function.resolver_function
def latest_created(artifacts, n: int = 1):
  """Returns the n latest createst artifacts, ties broken by artifact id.

  Args:
    artifacts: The artifacts to parse.
    n: The number of latest artifacts to return, must be > 0.

  Returns:
    The n latest artifacts.
  """
  return ops.LatestCreateTime(artifacts, n=n)


@resolver_function.resolver_function
def latest_pipeline_run(artifacts):
  """Returns the n latest createst artifacts, ties broken by artifact id.

  Args:
    artifacts: The artifacts to parse.

  Returns:
    The n latest artifacts.
  """
  return ops.LatestPipelineRun(artifacts)
