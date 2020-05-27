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
from typing import Any, Dict, Optional, Text, Type

from six import with_metaclass

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.components.base import executor_spec as executor_spec_module
from tfx.types import node_common
from tfx.utils import json_utils


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class BaseNode(with_metaclass(abc.ABCMeta, json_utils.Jsonable)):
  """Base class for a node in TFX pipeline."""

  @classmethod
  def get_id(cls, instance_name: Optional[Text] = None):
    """Gets the id of a node.

    This can be used during pipeline authoring time. For example:
    from tfx.components import Trainer

    resolver = ResolverNode(..., model=Channel(
        type=Model, producer_component_id=Trainer.get_id('my_trainer')))

    Args:
      instance_name: (Optional) instance name of a node. If given, the instance
        name will be taken into consideration when generating the id.

    Returns:
      an id for the node.
    """
    node_class_name = cls.__name__
    if instance_name:
      return '{}.{}'.format(node_class_name, instance_name)
    else:
      return node_class_name

  def __init__(
      self,
      instance_name: Optional[Text] = None,
      executor_spec: Optional[executor_spec_module.ExecutorSpec] = None,
      driver_class: Optional[Type[base_driver.BaseDriver]] = None,
  ):
    """Initialize a node.

    Args:
      instance_name: Optional unique identifying name for this instance of node
        in the pipeline. Required if two instances of the same node are used in
        the pipeline.
      executor_spec: Optional instance of executor_spec.ExecutorSpec which
        describes how to execute this node (optional, defaults to an empty
        executor indicates no-op.
      driver_class: Optional subclass of base_driver.BaseDriver as a custom
        driver for this node (optional, defaults to base_driver.BaseDriver).
        Nodes usually use the default driver class, but may override it.
    """
    if executor_spec is None:
      executor_spec = executor_spec_module.ExecutorClassSpec(
          base_executor.EmptyExecutor)
    if driver_class is None:
      driver_class = base_driver.BaseDriver
    self._instance_name = instance_name
    self.executor_spec = executor_spec
    self.driver_class = driver_class
    self._upstream_nodes = set()
    self._downstream_nodes = set()

  def to_json_dict(self) -> Dict[Text, Any]:
    """Convert from an object to a JSON serializable dictionary."""
    return dict((k, v)
                for k, v in self.__dict__.items()
                if k not in ['_upstream_nodes', '_downstream_nodes'])

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
    """Experimental: Add another component that must run before this one.

    This method enables task-based dependencies by enforcing execution order for
    synchronous pipelines on supported platforms. Currently, the supported
    platforms are Airflow, Beam, and Kubeflow Pipelines.

    Note that this API call should be considered experimental, and may not work
    with asynchronous pipelines, sub-pipelines and pipelines with conditional
    nodes. We also recommend relying on data for capturing dependencies where
    possible to ensure data lineage is fully captured within MLMD.

    It is symmetric with `add_downstream_node`.

    Args:
      upstream_node: a component that must run before this node.
    """
    self._upstream_nodes.add(upstream_node)
    if self not in upstream_node.downstream_nodes:
      upstream_node.add_downstream_node(self)

  @property
  def downstream_nodes(self):
    return self._downstream_nodes

  def add_downstream_node(self, downstream_node):
    """Experimental: Add another component that must run after this one.

    This method enables task-based dependencies by enforcing execution order for
    synchronous pipelines on supported platforms. Currently, the supported
    platforms are Airflow, Beam, and Kubeflow Pipelines.

    Note that this API call should be considered experimental, and may not work
    with asynchronous pipelines, sub-pipelines and pipelines with conditional
    nodes. We also recommend relying on data for capturing dependencies where
    possible to ensure data lineage is fully captured within MLMD.

    It is symmetric with `add_upstream_node`.

    Args:
      downstream_node: a component that must run after this node.
    """
    self._downstream_nodes.add(downstream_node)
    if self not in downstream_node.upstream_nodes:
      downstream_node.add_upstream_node(self)
