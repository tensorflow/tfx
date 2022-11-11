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
"""Definition and related classes for TFX pipeline."""

import copy
import enum
from typing import Any, Collection, Dict, List, Optional, Union, cast
import warnings

from tfx.dsl.compiler import constants
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.context_managers import dsl_context_registry
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import channel
from tfx.types import channel_utils
from tfx.utils import doc_controls
from tfx.utils import topsort

from google.protobuf import message

# Argo's workflow name cannot exceed 63 chars:
# see https://github.com/argoproj/argo/issues/1324.
# MySQL's database name cannot exceed 64 chars:
# https://dev.mysql.com/doc/refman/5.6/en/identifiers.html
_MAX_PIPELINE_NAME_LENGTH = 63

# Pipeline root is by default specified as a RuntimeParameter when runnning on
# KubeflowDagRunner. This constant offers users an easy access to the pipeline
# root placeholder when defining a pipeline. For example,
#
# pusher = Pusher(
#     model=trainer.outputs['model'],
#     model_blessing=evaluator.outputs['blessing'],
#     push_destination=pusher_pb2.PushDestination(
#         filesystem=pusher_pb2.PushDestination.Filesystem(
#             base_directory=os.path.join(
#                 str(pipeline.ROOT_PARAMETER), 'model_serving'))))
ROOT_PARAMETER = data_types.RuntimeParameter(
    name=constants.PIPELINE_ROOT_PARAMETER_NAME, ptype=str)


class ExecutionMode(enum.Enum):
  """Execution mode of a pipeline.

  Please see this
  [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200601-tfx-udsl-semantics.md)
  for more details.
  """
  SYNC = 1
  ASYNC = 2


def add_beam_pipeline_args_to_component(component, beam_pipeline_args):
  if isinstance(component.executor_spec, executor_spec.BeamExecutorSpec):
    # Prepend pipeline-level beam_pipeline_args in front of component specific
    # ones to make component-level override pipeline-level args.
    cast(
        executor_spec.BeamExecutorSpec,
        component.executor_spec).beam_pipeline_args = beam_pipeline_args + cast(
            executor_spec.BeamExecutorSpec,
            component.executor_spec).beam_pipeline_args


class PipelineInputs:
  """A utility class to help declare input signatures of composable pipelines."""

  def __init__(self, inputs: Optional[Dict[str, channel.BaseChannel]] = None):
    self._inputs = inputs or {}
    self._wrapped_inputs = {
        k: channel.PipelineInputChannel(v, output_key=k)
        for k, v in self._inputs.items()
    }
    self._pipeline = None

  @property
  def raw_inputs(self) -> Dict[str, channel.BaseChannel]:
    return self._inputs

  @property
  def inputs(self) -> Dict[str, channel.PipelineInputChannel]:
    return self._wrapped_inputs

  def __getitem__(self, key) -> channel.PipelineInputChannel:
    return self._wrapped_inputs[key]

  @property
  def pipeline(self) -> 'Pipeline':
    return self._pipeline

  @pipeline.setter
  def pipeline(self, pipeline: 'Pipeline'):
    self._pipeline = pipeline
    for c in self._wrapped_inputs.values():
      c.pipeline = pipeline


class RunOptions:
  r"""Run-time options for running a pipeline (such as partial run).

  To run a sub-graph of the Pipeline, include this when constructing the
  Pipeline object.

  ### Specifying the sub-graph to run

  To define the sub-graph to run, specify a set of *source nodes* and a set
  of *sink nodes*. These can be provided as collections of node_id strings, or
  as functions that takes a node id string and returns a boolean.

  Consider this pipeline graph:


                            -- CsvExampleGen --
                            |        |        |
                            v        |        |
               -- StatisticsGen      |        |
               |           |         |        |
               |           v         |        |
               |       SchemaGen     |        |
               |     /     |     \   |        |
               v    v      |      v  v        |
        ExampleValidator   |     Transform    |
                           |    /             |
                           v   v              |
                          Trainer -----       |
                              \        \      |
                               \        v     v
                                \      Evaluator
                                 \        /
                                  \      /
                                   v    v
                                   Pusher


  Suppose the user has already done a full pipeline run, and now only wants to
  run "Trainer" and "Evaluator". To specify this:

  ```python
  my_pipeline = pipeline.Pipeline(
      # How you'll normally define a pipeline.
      pipeline_name=...,
      # Include *all* pipeline components as usual, even the ones to be skipped.
      components=[
          example_gen_component,
          ...,
          pusher_component,
      ],
      # Add RunOptions to specify a partial run.
      run_options=pipeline.RunOptions(
          from_nodes=[trainer_component.id],
          to_nodes=[evaluator_component.id],
      ),
  )
  ```

  The compiler will find the nodes reachable downstream from Trainer and
  reachable upstream from Evaluator to obtain {Trainer, Evaluator}, and mark
  those nodes in the pipeline IR accordingly.

  Alternatively, the user can specify:

  ```python
  nodes_to_include = {trainer_component.id, evaluator_component.id}
  run_options = pipeline.RunOptions(
      from_nodes=nodes_to_include,
      to_nodes=nodes_to_include,
  )
  ```

  ### Specifying the artifact reuse strategy

  In the above example, the Trainer node is the first node in the partial run.
  How would it resolve its dependencies? By default, nodes in a partial run
  would use the output artifacts from the *previous pipeline run* (including
  partial runs) to resolve any dependencies that cannot be provided by other
  nodes in the same partial run.

  Using the previous pipeline run is sufficient in most cases. However,
  the user may wish to use a different pipeline run to provide missing
  dependencies -- perhaps an even earlier pipeline run. To specify this:

  ```python
  run_options = pipeline.RunOptions(
      from_nodes=...,
      to_nodes=...,
      base_pipeline_run_id=<the pipeline run id whose artifacts are to be used>,
  )
  ```
  """

  def __init__(self,
               from_nodes: Optional[Collection[str]] = None,
               to_nodes: Optional[Collection[str]] = None,
               base_pipeline_run_id: Optional[str] = None):
    """Constructor.

    Args:
      from_nodes: node_ids to be used as "from_nodes". Defaults to None,
        which indicates all nodes.
      to_nodes: node_ids to be used as "to_nodes". Defaults to None, which
        indicates all nodes.
      base_pipeline_run_id: If provided, will use this as the pipeline run id
        from which missing dependencies are provided. If None, will use the
        previous pipeline run id. Defaults to None.

    Raises:
      ValueError if both from_nodes or to_nodes are empty.
    """
    if not(from_nodes or to_nodes):
      raise ValueError('from_nodes or to_nodes cannot both be empty.')
    self.from_nodes = from_nodes
    self.to_nodes = to_nodes
    self.base_pipeline_run_id = base_pipeline_run_id


class Pipeline(base_node.BaseNode):
  """Logical TFX pipeline object.

  Pipeline object represents the DAG of TFX components, which can be run using
  one of the pipeline orchestration systems that TFX supports. For details,
  please refer to the
  [guide](https://github.com/tensorflow/tfx/blob/master/docs/guide/build_tfx_pipeline.md).

  Attributes:
    components: A deterministic list of logical components of this pipeline,
      which are deduped and topologically sorted.
    enable_cache: Whether or not cache is enabled for this run.
    metadata_connection_config: The config to connect to ML metadata.
    execution_mode: Execution mode of the pipeline. Currently only support
      synchronous execution mode.
    beam_pipeline_args: Pipeline arguments for Beam powered Components. Use
      `with_beam_pipeline_args` to set component level Beam args.
    platform_config: Pipeline level platform config, in proto form.
  """

  def __init__(self,
               pipeline_name: str,
               pipeline_root: Optional[Union[str, ph.Placeholder]] = '',
               metadata_connection_config: Optional[
                   metadata.ConnectionConfigType] = None,
               components: Optional[List[base_node.BaseNode]] = None,
               enable_cache: Optional[bool] = False,
               beam_pipeline_args: Optional[List[Union[str,
                                                       ph.Placeholder]]] = None,
               platform_config: Optional[message.Message] = None,
               execution_mode: Optional[ExecutionMode] = ExecutionMode.SYNC,
               inputs: Optional[PipelineInputs] = None,
               outputs: Optional[Dict[str, channel.OutputChannel]] = None,
               **kwargs):
    """Initialize pipeline.

    Args:
      pipeline_name: Name of the pipeline;
      pipeline_root: Path to root directory of the pipeline. This will most
        often be just a string. Some orchestrators may have limited support for
        constructing this from a Placeholder, e.g. a RuntimeInfoPlaceholder that
        refers to fields from the platform config. pipeline_root is optional
        only if the pipeline is composed within another parent pipeline, in
        which case it will inherit its parent pipeline's root.
      metadata_connection_config: The config to connect to ML metadata.
      components: Optional list of components to construct the pipeline.
      enable_cache: Whether or not cache is enabled for this run.
      beam_pipeline_args: Pipeline arguments for Beam powered Components.
      platform_config: Pipeline level platform config, in proto form.
      execution_mode: The execution mode of the pipeline, can be SYNC or ASYNC.
      inputs: Optional inputs of a pipeline.
      outputs: Optional outputs of a pipeline.
      **kwargs: Additional kwargs forwarded as pipeline args.
    """
    if len(pipeline_name) > _MAX_PIPELINE_NAME_LENGTH:
      raise ValueError(
          f'pipeline {pipeline_name} exceeds maximum allowed length: {_MAX_PIPELINE_NAME_LENGTH}.'
      )
    self.pipeline_name = pipeline_name

    # Initialize pipeline as a node.
    super().__init__()

    if inputs:
      inputs.pipeline = self
    self._inputs = inputs
    if outputs:
      self._outputs = {
          k: channel.PipelineOutputChannel(v, pipeline=self, output_key=k)
          for k, v in outputs.items()
      }
    else:
      self._outputs = {}
    self._id = pipeline_name

    # Once pipeline is finalized, this instance is regarded as immutable and
    # any detectable mutation will raise an error.
    self._finalized = False

    # TODO(b/183621450): deprecate PipelineInfo.
    self.pipeline_info = data_types.PipelineInfo(  # pylint: disable=g-missing-from-attributes
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root)
    self.enable_cache = enable_cache
    self.metadata_connection_config = metadata_connection_config
    self.execution_mode = execution_mode

    self._beam_pipeline_args = beam_pipeline_args or []

    self.platform_config = platform_config

    self.additional_pipeline_args = kwargs.pop(  # pylint: disable=g-missing-from-attributes
        'additional_pipeline_args', {})

    reg = kwargs.pop('dsl_context_registry', None)
    if reg:
      if not isinstance(reg, dsl_context_registry.DslContextRegistry):
        raise ValueError('dsl_context_registry must be DslContextRegistry type '
                         f'but got {reg}')
      self._dsl_context_registry = reg
    else:
      self._dsl_context_registry = dsl_context_registry.get()
      if self._dsl_context_registry.get_contexts(self):
        self._dsl_context_registry = (
            self._dsl_context_registry.extract_for_pipeline(self))

    # TODO(b/216581002): Use self._dsl_context_registry to obtain components.
    self._components = []
    if components:
      self._set_components(components)

  def _check_mutable(self):
    if self._finalized:
      raise RuntimeError('Cannot mutate Pipeline after finalize.')

  @property
  def beam_pipeline_args(self):
    """Beam pipeline args used for all components in the pipeline."""
    return self._beam_pipeline_args

  @property
  @doc_controls.do_not_generate_docs
  def dsl_context_registry(self) -> dsl_context_registry.DslContextRegistry:  # pylint: disable=g-missing-from-attributes
    if self._dsl_context_registry is None:
      raise RuntimeError('DslContextRegistry is not persisted yet. Run '
                         'pipeline.finalize() first.')
    return self._dsl_context_registry

  @property
  def components(self):
    """A deterministic list of logical components that are deduped and topologically sorted."""
    return self._components

  @components.setter
  def components(self, components: List[base_node.BaseNode]):
    self._set_components(components)

  def _set_components(self, components: List[base_node.BaseNode]) -> None:
    """Set a full list of components of the pipeline."""
    self._check_mutable()

    deduped_components = set(components)
    node_by_id = {}

    # Fills in producer map.
    for component in deduped_components:
      # Checks every node has an unique id.
      if component.id in node_by_id:
        raise RuntimeError(
            f'Duplicated node_id {component.id} for component type'
            f'{component.type}.')
      node_by_id[component.id] = component

    # Connects nodes based on producer map.
    for component in deduped_components:
      channels = list(component.inputs.values())
      for exec_property in component.exec_properties.values():
        if isinstance(exec_property, ph.ChannelWrappedPlaceholder):
          channels.append(exec_property.channel)
      for predicate in conditional.get_predicates(component,
                                                  self.dsl_context_registry):
        for chnl in predicate.dependent_channels():
          channels.append(chnl)

      for input_channel in channels:
        for node_id in channel_utils.get_dependent_node_ids(input_channel):
          if node_id == self.id:
            # If a component's input channel depends on the (self) pipeline,
            # it means that component consumes pipeline-level inputs. No need to
            # add upstream node here. Pipeline-level inputs will be handled
            # during compilation.
            continue
          upstream_node = node_by_id.get(node_id)
          if upstream_node:
            component.add_upstream_node(upstream_node)
            upstream_node.add_downstream_node(component)
          else:
            warnings.warn(
                f'Node {component.id} depends on the output of node {node_id}'
                f', but {node_id} is not included in the components of '
                'pipeline. Did you forget to add it?')

    layers = topsort.topsorted_layers(
        list(deduped_components),
        get_node_id_fn=lambda c: c.id,
        get_parent_nodes=lambda c: c.upstream_nodes,
        get_child_nodes=lambda c: c.downstream_nodes)
    self._components = []
    for layer in layers:
      for component in layer:
        self._components.append(component)

    if self.beam_pipeline_args:
      for component in self._components:
        add_beam_pipeline_args_to_component(component, self.beam_pipeline_args)

  @doc_controls.do_not_generate_docs
  def finalize(self):
    self._persist_dsl_context_registry()
    self._finalized = True

  def _persist_dsl_context_registry(self):
    """Persist the DslContextRegistry to the pipeline."""
    self._dsl_context_registry = copy.copy(self._dsl_context_registry)
    self._dsl_context_registry.finalize()

    given_components = set(self._components)
    registry_components = set(self._dsl_context_registry.all_nodes)
    for unseen_component in given_components - registry_components:
      warnings.warn(
          f'Component {unseen_component.id} is not found from the registry. '
          'This is probably due to reusing component from another pipeline '
          'or interleaved pipeline definitions. Make sure each component '
          'belong to exactly one pipeline, and pipeline definitions are '
          'separated.')

  @property
  def inputs(self) -> Dict[str, Any]:
    # If we view a Pipeline as a Node, its inputs should be unwrapped (raw)
    # channels that are provided through PipelineInputs, and consumed by nodes
    # in the inner pipeline.
    if self._inputs:
      return self._inputs.raw_inputs
    else:
      return {}

  @property
  def outputs(self) -> Dict[str, Any]:
    # If we view a Pipeline as a Node, its outputs should be wrapped channels
    # that will be consumed by nodes in the outer pipeline.
    return self._outputs

  @property
  def exec_properties(self) -> Dict[str, Any]:
    return {}
