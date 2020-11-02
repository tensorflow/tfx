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
"""Representations of ResolverConfig.Graph executions."""

import collections.abc
import copy
import itertools
from typing import List, Mapping, Text, Collection, Sequence, Dict

from tfx.orchestration.experimental.resolver import operator
from tfx.proto.orchestration import pipeline_pb2

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_MlmdArtfiact = metadata_store_pb2.Artifact


class GraphRun:
  """Run of ResolverConfig.ResolverGraphDef.

  Graph will be run during instance initialization (`__init__`).
  """

  def __init__(self, graph_def: pipeline_pb2.ResolverConfig.ResolverGraphDef,
               metadata_store: mlmd.MetadataStore,
               channel_inputs: Mapping[Text, Collection[_MlmdArtfiact]]):
    self._memoized_output = {}
    self._nodes = {node.node_id: node for node in graph_def.nodes}
    self._graph_def = graph_def
    self._context = operator.OperatorRunContext(
        store=metadata_store,
        channel_inputs=channel_inputs)
    self._multi_trigger_key = None
    self._sufficient = True

    # Execution and validation will happen on the __init__ of the instance.
    self._run_result = self._run()
    self._validate_run_result()

  def resolved_mlmd_artifacts(self) \
      -> List[Dict[Text, Sequence[_MlmdArtfiact]]]:
    """Get graph run result in List[mlmd_input_map] format."""
    if not self._sufficient:
      return []
    # All result has at least one triggering inputs.
    base_input_map = {
        key: triggers[0]
        for key, triggers in self._run_result.items()
        if key != self._multi_trigger_key
    }
    if self._multi_trigger_key is None:
      return [base_input_map]
    result = []
    for inputs in self._run_result[self._multi_trigger_key]:
      input_map = copy.deepcopy(base_input_map)
      input_map[self._multi_trigger_key] = inputs
      result.append(input_map)
    return result

  def _validate_run_result(self):
    """Check the graph run result conforms to the input resolution result."""
    result_type_error = RuntimeError(
        'Input resolution result should be Sequence[Sequence[Artifact]]. '
        'Got {0!r}'.format(self._run_result))
    for key, triggers in self._run_result.items():
      if not isinstance(triggers, collections.abc.Sequence):
        raise result_type_error
      if len(triggers) == 0:  # pylint: disable=g-explicit-length-test
        self._sufficient = False
        continue
      elif len(triggers) > 1:
        if self._multi_trigger_key is not None:
          raise RuntimeError(
              'At most one input can have multiple triggering inputs. Both '
              '{0} and {1} have multiple triggering inputs'
              .format(key, self._multi_trigger_key))
        self._multi_trigger_key = key
      for inputs in triggers:
        if not isinstance(inputs, collections.abc.Sequence):
          raise result_type_error
        if not all(isinstance(a, _MlmdArtfiact) for a in inputs):
          raise result_type_error
      artifact_type_ids = set(
          a.type_id for a in itertools.chain.from_iterable(triggers))
      if len(artifact_type_ids) != 1:
        raise RuntimeError(
            'Each channel should have a homogeneous artifact type; Got {}.'
            .format(artifact_type_ids))

  def _run(self):
    result = {}
    for label, node_id in self._graph_def.sink.items():
      result[label] = self._get_node_output(node_id)
    return result

  def _get_node_output(self, node_id):
    if node_id not in self._memoized_output:
      self._memoized_output[node_id] = self._run_node(node_id)
    return self._memoized_output[node_id]

  def _run_node(self, node_id):
    """Run a single operator node."""
    node = self._nodes.get(node_id)
    if node is None:
      raise RuntimeError('Could not find node_id={} in ResolverConfig IR.'
                         .format(node_id))
    op = operator.OpsRegistry.get(node.op_name)
    kwargs = {
        name: self._prepare_node_input(node_input)
        for name, node_input in node.named_inputs.items()
    }
    return op.run(self._context, **kwargs)

  def _prepare_node_input(
      self, node_input: pipeline_pb2.ResolverConfig.NodeInput):
    """Prepare single NodeInput, which may need to run other nodes."""
    input_type = node_input.WhichOneof('input_type')
    if not input_type:
      raise ValueError('Invalid IR proto')
    if input_type in ('int_value', 'string_value', 'proto_value'):
      return getattr(node_input, input_type)
    elif input_type == 'input_list':
      return [self._prepare_node_input(item)
              for item in node_input.input_list.values]
    elif input_type == 'input_map':
      return {key: self._prepare_node_input(value)
              for key, value in node_input.input_map.values.items()}
    elif input_type == 'node_output':
      return self._get_node_output(node_input.node_output)
    else:
      # Should not reach here.
      raise NotImplementedError('Unknown input_type {}'.format(input_type))
