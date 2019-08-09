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

from six import with_metaclass

from typing import Any, Dict, Optional, Text, Type

from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.types import component_spec


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class BaseComponent(with_metaclass(abc.ABCMeta, object)):
  """Base class for a TFX pipeline component.

  An instance of a subclass of BaseComponent represents the parameters for a
  single execution of that TFX pipeline component.

  All subclasses of BaseComponent must override the SPEC_CLASS field with the
  ComponentSpec subclass that defines the interface of this component.

  Attributes:
    COMPONENT_NAME: the name of this component, as a string (optional). If not
      specified, the name of the component class will be used.
    SPEC_CLASS: a subclass of types.ComponentSpec used by this component
      (required).
    EXECUTOR_CLASS: a subclass of base_executor.BaseExecutor used to execute
      this component (required).
    DRIVER_CLASS: a subclass of base_driver.BaseDriver as a custom driver for
      this component (optional, defaults to base_driver.BaseDriver).
  """

  # Subclasses may optionally override this component name property, which
  # would otherwise default to the component class name. This is used for
  # tracking component instances in pipelines and display in the UI.
  COMPONENT_NAME = None
  # Subclasses must override this property (by specifying a types.ComponentSpec
  # class, e.g. "SPEC_CLASS = MyComponentSpec").
  SPEC_CLASS = _abstract_property()
  # Subclasses must also override the executor class.
  EXECUTOR_CLASS = _abstract_property()
  # Subclasses will usually use the default driver class, but may override this
  # property as well.
  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(
      self,
      spec: types.ComponentSpec,
      custom_executor_class: Optional[Type[base_executor.BaseExecutor]] = None,
      name: Optional[Text] = None,
      component_name: Optional[Text] = None):
    """Initialize a component.

    Args:
      spec: types.ComponentSpec object for this component instance.
      custom_executor_class: Optional custom executor class overriding the
        default executor specified in the component attribute.
      name: Optional unique identifying name for this instance of the component
        in the pipeline. Required if two instances of the same component is used
        in the pipeline.
      component_name: Optional component name override for this instance of the
        component. This will override the class-level COMPONENT_NAME attribute
        and the name of the class (which would otherwise be used as the
        component name).
    """
    self.spec = spec
    if custom_executor_class:
      if not issubclass(custom_executor_class, base_executor.BaseExecutor):
        raise TypeError(
            ('Custom executor class override %s for %s should be a subclass of '
             'base_executor.BaseExecutor') %
            (custom_executor_class, self.__class__))
    self.executor_class = (
        custom_executor_class or self.__class__.EXECUTOR_CLASS)
    self.driver_class = self.__class__.DRIVER_CLASS
    self.component_name = (component_name
                           or self.__class__.COMPONENT_NAME
                           or self.__class__.__name__)
    self.name = name
    self._upstream_nodes = set()
    self._downstream_nodes = set()
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
    if not (inspect.isclass(cls.EXECUTOR_CLASS) and
            issubclass(cls.EXECUTOR_CLASS, base_executor.BaseExecutor)):
      raise TypeError((
          'Component class %s expects EXECUTOR_CLASS property to be a subclass '
          'of base_executor.BaseExecutor; got %s instead.') %
                      (cls, cls.EXECUTOR_CLASS))
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
    return ('%s(spec: %s, executor_class: %s, driver_class: %s, name: %s, '
            'inputs: %s, outputs: %s)') % (
                self.__class__.__name__, self.spec, self.executor_class,
                self.driver_class, self.name, self.inputs, self.outputs)

  @property
  def component_type(self) -> Text:
    return '.'.join([self.__class__.__module__, self.__class__.__name__])

  @property
  def inputs(self) -> component_spec._PropertyDictWrapper:
    return self.spec.inputs

  @property
  def outputs(self) -> component_spec._PropertyDictWrapper:
    return self.spec.outputs

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return self.spec.exec_properties

  # TODO(ruoyu): Consolidate the usage of component identifier. Moving forward,
  # we will have two component level keys:
  # - component_type: the path of the python executor or the image uri of the
  #   executor.
  # - component_id: <component_name>.<unique_name>
  @property
  def component_id(self):
    """Component id.

    If unique name is available, component_id will be:
      <component_name>.<unique_name>
    otherwise, component_id will be:
      <component_name>

    Returns:
      component id.
    """
    if self.name:
      return '{}.{}'.format(self.component_name, self.name)
    else:
      return self.component_name

  @property
  def upstream_nodes(self):
    return self._upstream_nodes

  def add_upstream_node(self, upstream_node):
    self._upstream_nodes.add(upstream_node)

  @property
  def downstream_nodes(self):
    return self._downstream_nodes

  def add_downstream_node(self, downstream_node):
    self._downstream_nodes.add(downstream_node)
