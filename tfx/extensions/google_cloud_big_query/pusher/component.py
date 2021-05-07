# Copyright 2021 Google LLC. All Rights Reserved.
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
"""TFX Pusher for pushing models to cloud big query."""

from typing import Any, Dict, Optional

from tfx import types
from tfx.components.pusher import component as pusher_component
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_big_query.pusher import executor


class Pusher(pusher_component.Pusher):
  """Cloud Big Query Pusher component.

  Component `outputs` contains:
   - `pushed_model`: Channel of type `standard_artifacts.PushedModel` with
                     result of push.
  """

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               model: Optional[types.Channel] = None,
               model_blessing: Optional[types.Channel] = None,
               infra_blessing: Optional[types.Channel] = None,
               custom_config: Optional[Dict[str, Any]] = None):
    """Construct a Pusher component.

    Args:
      model: An optional Channel of type `standard_artifacts.Model`, usually
        produced by a Trainer component.
      model_blessing: An optional Channel of type
        `standard_artifacts.ModelBlessing`, usually produced from an Evaluator
        component.
      infra_blessing: An optional Channel of type
        `standard_artifacts.InfraBlessing`, usually produced from an
        InfraValidator component.
      custom_config: A dict which contains the deployment job parameters to be
        passed to Cloud platforms.
    """
    super(Pusher, self).__init__(
        model=model,
        model_blessing=model_blessing,
        infra_blessing=infra_blessing,
        custom_config=custom_config)
