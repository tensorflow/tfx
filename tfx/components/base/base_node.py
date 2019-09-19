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

from six import with_metaclass

from typing import Any, Optional, Text

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.types import node_common


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class BaseNode(with_metaclass(abc.ABCMeta, object)):
  """Base class for a node in TFX pipeline DAG."""

  def __init__(self, instance_name: Optional[Text] = None):
    self._instance_name = instance_name
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
  def inputs(self) -> node_common._PropertyDictWrapper:
    pass

  @property
  @abc.abstractmethod
  def outputs(self) -> node_common._PropertyDictWrapper:
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
