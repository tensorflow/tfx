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
"""Definition of TFX runner base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
from six import with_metaclass
from typing import Any, Optional, Type

from tfx.components.base import base_component
from tfx.orchestration.launcher import base_component_launcher
from tfx.utils import abc_utils


class TfxRunner(with_metaclass(abc.ABCMeta, object)):
  """Base runner class for TFX.

  This is the base class for every TFX runner.
  """

  # A list of component launcher classes that are supported by the current
  # runner. List sequence determines the order in which launchers are chosen
  # for each component being run.
  # Subclasses must override this property by specifying a list of supported
  # launcher classes, e.g.
  # `SUPPORTED_LAUNCHER_CLASSES = [InProcessComponentLauncher]`.
  SUPPORTED_LAUNCHER_CLASSES = abc_utils.abstract_property()

  def __init__(self):
    """Initializes a TfxRunner instance.
    """
    self._supported_launcher_classes = self.__class__.SUPPORTED_LAUNCHER_CLASSES
    self._validate_supported_launcher_classes()

  def _validate_supported_launcher_classes(self):
    if not self._supported_launcher_classes:
      raise ValueError('component_launcher_classes must not be None or empty.')

    if any([
        not issubclass(cls, base_component_launcher.BaseComponentLauncher)
        for cls in self._supported_launcher_classes
    ]):
      raise TypeError('Each item in supported_launcher_classes must be type of '
                      'base_component_launcher.BaseComponentLauncher.')

  def find_component_launcher_class(
      self, component: base_component.BaseComponent
  ) -> Type[base_component_launcher.BaseComponentLauncher]:
    """Find a launcher in the runner which can launch the component.

    The default lookup logic goes through the self._supported_launcher_classes
    in order and returns the first one which can launch the executor_spec of
    the component. Subclass may customize the logic by overriding the method.

    Args:
      component: the component to launch.

    Returns:
      The found component launcher for the component.

    Raises:
      RuntimeError: if no supported launcher is found.
    """
    for component_launcher_class in self._supported_launcher_classes:
      if component_launcher_class.can_launch(component.executor_spec):
        return component_launcher_class
    raise RuntimeError('No launcher can launch component "%s".' %
                       component.component_id)

  @abc.abstractmethod
  def run(self, pipeline) -> Optional[Any]:
    """Runs logical TFX pipeline on specific platform.

    Args:
      pipeline: logical TFX pipeline definition.

    Returns:
      Platform-specific object.
    """
    pass
