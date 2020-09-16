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
"""Base class for TFX components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
from typing import Any, Dict, Optional, Text

from six import with_metaclass

from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.types import node_common
from tfx.utils import abc_utils

# Constants that used for serializing and de-serializing components.
_DRIVER_CLASS_KEY = 'driver_class'
_EXECUTOR_SPEC_KEY = 'executor_spec'
_INSTANCE_NAME_KEY = '_instance_name'
_SPEC_KEY = 'spec'


class BaseComponent(with_metaclass(abc.ABCMeta, base_node.BaseNode)):
  """Base class for a TFX pipeline component.

  An instance of a subclass of BaseComponent represents the parameters for a
  single execution of that TFX pipeline component.

  All subclasses of BaseComponent must override the SPEC_CLASS field with the
  ComponentSpec subclass that defines the interface of this component.

  Attributes:
    SPEC_CLASS: a subclass of types.ComponentSpec used by this component
      (required).
    EXECUTOR_SPEC: an instance of executor_spec.ExecutorSpec which describes how
      to execute this component (required).
    DRIVER_CLASS: a subclass of base_driver.BaseDriver as a custom driver for
      this component (optional, defaults to base_driver.BaseDriver).
  """

  # Subclasses must override this property (by specifying a types.ComponentSpec
  # class, e.g. "SPEC_CLASS = MyComponentSpec").
  SPEC_CLASS = abc_utils.abstract_property()
  # Subclasses must also override the executor spec.
  #
  # Note: EXECUTOR_CLASS has been replaced with EXECUTOR_SPEC. A custom
  # component's existing executor class definition "EXECUTOR_CLASS = MyExecutor"
  # should be replaced with "EXECUTOR_SPEC = ExecutorClassSpec(MyExecutor).
  EXECUTOR_SPEC = abc_utils.abstract_property()
  # Subclasses will usually use the default driver class, but may override this
  # property as well.
  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(
      self,
      spec: types.ComponentSpec,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      instance_name: Optional[Text] = None):
    """Initialize a component.

    Args:
      spec: types.ComponentSpec object for this component instance.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor specified in the component attribute.
      instance_name: Optional unique identifying name for this instance of the
        component in the pipeline. Required if two instances of the same
        component is used in the pipeline.
    """
    executor_spec_obj = (custom_executor_spec or self.__class__.EXECUTOR_SPEC)
    driver_class = self.__class__.DRIVER_CLASS
    super(BaseComponent, self).__init__(
        instance_name=instance_name,
        executor_spec=executor_spec_obj,
        driver_class=driver_class,
    )
    self.spec = spec
    if custom_executor_spec:
      if not isinstance(custom_executor_spec, executor_spec.ExecutorSpec):
        raise TypeError(
            ('Custom executor spec override %s for %s should be an instance of '
             'ExecutorSpec') % (custom_executor_spec, self.__class__))
    self._validate_component_class()
    self._validate_spec(spec)

  @classmethod
  def _validate_component_class(cls):
    """Validate that the SPEC_CLASSES property of this class is set properly."""
    if not (inspect.isclass(cls.SPEC_CLASS) and
            issubclass(cls.SPEC_CLASS, types.ComponentSpec)):
      raise TypeError(
          ('Component class %s expects SPEC_CLASS property to be a subclass '
           'of types.ComponentSpec; got %s instead.') % (cls, cls.SPEC_CLASS))
    if not isinstance(cls.EXECUTOR_SPEC, executor_spec.ExecutorSpec):
      raise TypeError((
          'Component class %s expects EXECUTOR_SPEC property to be an instance '
          'of ExecutorSpec; got %s instead.') % (cls, type(cls.EXECUTOR_SPEC)))
    if not (inspect.isclass(cls.DRIVER_CLASS) and
            issubclass(cls.DRIVER_CLASS, base_driver.BaseDriver)):
      raise TypeError(
          ('Component class %s expects DRIVER_CLASS property to be a subclass '
           'of base_driver.BaseDriver; got %s instead.') %
          (cls, cls.DRIVER_CLASS))

  def _validate_spec(self, spec):
    """Verify given spec is valid given the component's SPEC_CLASS."""
    if not isinstance(spec, types.ComponentSpec):
      raise ValueError((
          'BaseComponent (parent class of %s) expects "spec" argument to be an '
          'instance of types.ComponentSpec, got %s instead.') %
                       (self.__class__, spec))
    if not isinstance(spec, self.__class__.SPEC_CLASS):
      raise ValueError(
          ('%s expects the "spec" argument to be an instance of %s; '
           'got %s instead.') %
          (self.__class__, self.__class__.SPEC_CLASS, spec))

  def __repr__(self):
    return ('%s(spec: %s, executor_spec: %s, driver_class: %s, '
            'component_id: %s, inputs: %s, outputs: %s)') % (
                self.__class__.__name__, self.spec, self.executor_spec,
                self.driver_class, self.id, self.inputs, self.outputs)

  @property
  def inputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return self.spec.inputs

  @property
  def outputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return self.spec.outputs

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return self.spec.exec_properties
