# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Operator and registry for the input resolver."""

from typing import Mapping, Text, Collection

import attr

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_MlmdArtfiact = metadata_store_pb2.Artifact


@attr.s(auto_attribs=True, kw_only=True, frozen=True)
class OperatorRunContext:
  store: mlmd.MetadataStore
  channel_inputs: Mapping[Text, Collection[_MlmdArtfiact]]

  def call(self, name, **kwargs):
    return OpsRegistry.get(name).run(self, **kwargs)


class Operator:
  """Operator for resolver.

  Input resolution for the TFX pipeline component is done by the computation
  graph (ResolverConfig.GraphDef), where operator is the node of the graph, and
  any type of python object can be produced or consumed from each operator node.

  Each operator instance corresponds to the different operator definition.
  Operator is identified by its name and cannot be overloaded.

  Operator is implemented by a python function that accepts "context" as the
  first argument and the named arguments. There's currently no type checking
  of arguments.

  Depending on the platform, different implementation for the same operator
  (name) can be registered in OpsRegistry and used.
  """

  def __init__(self, fn):
    self._impl = fn

  @property
  def name(self):
    return self._impl.__name__

  def run(self, context: OperatorRunContext, **kwargs):
    return self._impl(context, **kwargs)


class OpsRegistry:
  """Singleton operator registry by its name."""

  _ops = {}

  @classmethod
  def register(cls, operator: Operator):
    if operator.name not in cls._ops:
      cls._ops[operator.name] = operator

  @classmethod
  def get(cls, name) -> Operator:
    return cls._ops[name]


def operator_function(fn):
  """Decorator for the operator implementation function."""
  op = Operator(fn)
  OpsRegistry.register(op)
  return op
