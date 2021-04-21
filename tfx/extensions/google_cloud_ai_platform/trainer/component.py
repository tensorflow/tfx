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
"""TFX Trainer for training model on AI Platform."""

from tfx.components.trainer import component as trainer_component
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.trainer import executor


class Trainer(trainer_component.Trainer):
  """Cloud AI Platform Trainer component."""

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.GenericExecutor)
