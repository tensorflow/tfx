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
"""Base class for TFX nodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any, Dict, Optional, Text

from six import with_metaclass

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.types import node_common
from tfx.utils import json_utils


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class BaseNode(with_metaclass(abc.ABCMeta, json_utils.Jsonable)):
  """Base class for a node in TFX pipeline.

  Attributes:
    EXECUTOR_SPEC: an instance of executor_spec.ExecutorSpec which describes how
      to execute this node (optional, defaults to an empty executor indicates
      no-op.
    DRIVER_CLASS: a subclass of base_driver.BaseDriver as a custom driver for
      this node (optional, defaults to base_driver.BaseDriver).
  """

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.EmptyExecutor)
  # Subclasses will usually use the default driver class, but may override this
  # property as well.
  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(self, instance_name: Optional[Text] = None):
    self._instance_name = instance_name
    self.executor_spec = self.__class__.EXECUTOR_SPEC
    self.driver_class = self.__class__.DRIVER_CLASS
    self._upstream_nodes = set()
    self._downstream_nodes = set()

  @property
  def type(self) -> Text:
    return '.'.join([self.__class__.__module__, self.__class__.__name__])

  @property
  @deprecation.deprecated(None,
                          'component_type is deprecated, use type instead')
  def component_type(self) -> Text:
    return self.type

  @property
  def id(self) -> Text:
    """Node id, unique across all TFX nodes in a pipeline.

    If instance name is available, node_id will be:
      <node_class_name>.<instance_name>
    otherwise, node_id will be:
      <node_class_name>

    Returns:
      node id.
    """
    node_class_name = self.__class__.__name__
    if self._instance_name:
      return '{}.{}'.format(node_class_name, self._instance_name)
    else:
      return node_class_name

  @property
  @deprecation.deprecated(None, 'component_id is deprecated, use id instead')
  def component_id(self) -> Text:
    return self.id

  @property
  @abc.abstractmethod
  def inputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    pass

  @property
  @abc.abstractmethod
  def outputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    pass

  @property
  @abc.abstractmethod
  def exec_properties(self) -> Dict[Text, Any]:
    pass

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
