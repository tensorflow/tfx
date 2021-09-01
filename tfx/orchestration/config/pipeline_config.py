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
"""Settings for controlling how to run a pipeline."""

from typing import Dict, List, Optional, Type

from tfx.orchestration.config import base_component_config
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher


class PipelineConfig:
  """Config class which controls how to run a pipeline.

  Attributes:
    supported_launcher_classes: A list of component launcher classes that are
      supported by the current pipeline. List sequence determines the order in
      which launchers are chosen for each component being run.
    default_component_configs: A list of default component configs which will
      be used as default component config to run each component in the pipeline.
      List sequence determines the order in which config are chosen for each
      component being run.
    component_config_overrides: component configs for customizing the launching
      of each component. The key is the component ID.
  """

  # TODO(hongyes): figure out the best practice to put the
  # SUPPORTED_LAUNCHER_CLASSES.
  def __init__(self,
               supported_launcher_classes: Optional[List[Type[
                   base_component_launcher.BaseComponentLauncher]]] = None,
               default_component_configs: Optional[List[
                   base_component_config.BaseComponentConfig]] = None,
               component_config_overrides: Optional[Dict[
                   str, base_component_config.BaseComponentConfig]] = None):
    self.supported_launcher_classes = supported_launcher_classes or [
        in_process_component_launcher.InProcessComponentLauncher
    ]
    self.default_component_configs = default_component_configs or []
    self.component_config_overrides = component_config_overrides or {}
    self._validate_configs()

  def _validate_configs(self):
    """Validate the config settings."""
    if len(self.supported_launcher_classes) > len(
        set(self.supported_launcher_classes)):
      raise ValueError(
          'supported_launcher_classes must not have duplicate types')
    default_component_config_classes = [
        type(config) for config in self.default_component_configs
    ]
    if len(default_component_config_classes) > len(
        set(default_component_config_classes)):
      raise ValueError(
          'default_component_configs must not have configs with the same type')
