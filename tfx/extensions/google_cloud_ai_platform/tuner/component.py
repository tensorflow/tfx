# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX Tuner that distributes tuner workers as a job on AI Platform Training."""

from tfx.components.tuner import component as tuner_component
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.tuner import executor


class Tuner(tuner_component.Tuner):
  """TFX component for model hyperparameter tuning on AI Platform Training."""

  # TODO(b/160260359): Decide if custom_executor_spec should be added to
  #                    TunerSpec, or deprecate other use of custom_executor_spec
  #                    and the interface to swap Executor for a component
  #                    entirely, to standarize around custom components.
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)
