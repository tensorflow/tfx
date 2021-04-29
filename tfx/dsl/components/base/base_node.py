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
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec as executor_spec_module
from tfx.types import node_common
from tfx.utils import deprecation_utils
from tfx.utils import doc_controls
from tfx.utils import json_utils


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class BaseNode(with_metaclass(abc.ABCMeta, json_utils.Jsonable)):
  """Base class for a node in TFX pipeline."""

  def __init__(
      self,
      executor_spec: Optional[executor_spec_module.ExecutorSpec] = None,
      driver_class: Optional[Type[base_driver.BaseDriver]] = None,
  ):
    """Initialize a node.

    Args:
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
    self.executor_spec = executor_spec
    self.driver_class = driver_class
    self._upstream_nodes = set()
    self._downstream_nodes = set()
    self._id = None

  @doc_controls.do_not_doc_in_subclasses
  def to_json_dict(self) -> Dict[Text, Any]:
    """Convert from an object to a JSON serializable dictionary."""
    return dict((k, v)
                for k, v in self.__dict__.items()
                if k not in ['_upstream_nodes', '_downstream_nodes'])

  @classmethod
  @doc_controls.do_not_doc_in_subclasses
  def get_class_type(cls) -> Text:
    nondeprecated_class = deprecation_utils.get_first_nondeprecated_class(cls)
    return '.'.join(
        [nondeprecated_class.__module__, nondeprecated_class.__name__])

  @property
  @doc_controls.do_not_doc_in_subclasses
  def type(self) -> Text:
    return self.__class__.get_class_type()

  @property
  @deprecation_utils.deprecated(None,
                                'component_type is deprecated, use type instead'
                               )
  @doc_controls.do_not_doc_in_subclasses
  def component_type(self) -> Text:
    return self.type

  @property
  @doc_controls.do_not_doc_in_subclasses
  def id(self) -> Text:
    """Node id, unique across all TFX nodes in a pipeline.

    If `id` is set by the user, return it directly.
    Otherwise, return <node_class_name>.

    Returns:
      node id.
    """
    if self._id:
      return self._id
    node_class = deprecation_utils.get_first_nondeprecated_class(self.__class__)
    return node_class.__name__

  @property
  @deprecation_utils.deprecated(None,
                                'component_id is deprecated, use id instead')
  @doc_controls.do_not_doc_in_subclasses
  def component_id(self) -> Text:
    return self.id

  @id.setter
  @doc_controls.do_not_doc_in_subclasses
  def id(self, id: Text) -> None:  # pylint: disable=redefined-builtin
    self._id = id

  @doc_controls.do_not_doc_in_subclasses
  def with_id(self, id: Text) -> 'BaseNode':  # pylint: disable=redefined-builtin
    self._id = id
    return self

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
  @doc_controls.do_not_doc_in_subclasses
  def upstream_nodes(self):
    return self._upstream_nodes

  @doc_controls.do_not_doc_in_subclasses
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
  @doc_controls.do_not_doc_in_subclasses
  def downstream_nodes(self):
    return self._downstream_nodes

  @doc_controls.do_not_doc_in_subclasses
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
