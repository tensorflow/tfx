# Lint as: python3
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
"""Utilities for handling common config operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Tuple, Type

from tfx.components.base import base_component
from tfx.orchestration.config import base_component_config
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import base_component_launcher


def find_component_launch_info(
    p_config: pipeline_config.PipelineConfig,
    component: base_component.BaseComponent,
) -> Tuple[Type[base_component_launcher.BaseComponentLauncher],
           Optional[base_component_config.BaseComponentConfig]]:
  """Find a launcher and component config to launch the component.

  The default lookup logic goes through the `supported_launcher_classes`
  in sequence for each config from the `default_component_configs`. User can
  override a single component setting by `component_config_overrides`. The
  method returns the first component config and launcher which together can
  launch the executor_spec of the component.
  Subclass may customize the logic by overriding the method.

  Args:
    p_config: the pipeline config.
    component: the component to launch.

  Returns:
    The found tuple of component launcher class and the compatible component
    config.

  Raises:
    RuntimeError: if no supported launcher is found.
  """
  if component.id in p_config.component_config_overrides:
    component_configs = [p_config.component_config_overrides[component.id]]
  else:
    # Add None to the end of the list to find launcher with no component
    # config
    component_configs = p_config.default_component_configs + [None]

  for component_config in component_configs:
    for component_launcher_class in p_config.supported_launcher_classes:
      if component_launcher_class.can_launch(component.executor_spec,
                                             component_config):
        return (component_launcher_class, component_config)
  raise RuntimeError('No launcher info can be found for component "%s".' %
                     component.component_id)
