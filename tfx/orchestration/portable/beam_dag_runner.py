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
"""Definition of Beam TFX runner."""

import datetime
import os
from typing import Any, Iterable, Optional, Union

from absl import logging
import apache_beam as beam
from tfx.dsl.compiler import compiler
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as pipeline_py
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable import tfx_runner
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import telemetry_utils

from google.protobuf import any_pb2
from google.protobuf import message


# TODO(jyzhao): confirm it's re-executable, add test case.
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(Any)
class PipelineNodeAsDoFn(beam.DoFn):
  """Wrap node as beam DoFn."""

  def __init__(self, pipeline_node: pipeline_pb2.PipelineNode,
               mlmd_connection_config: metadata.ConnectionConfigType,
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
               executor_spec: Optional[message.Message],
               custom_driver_spec: Optional[message.Message],
               deployment_config: Optional[message.Message]):
    """Initializes the PipelineNodeAsDoFn.

    Args:
      pipeline_node: The specification of the node that this launcher lauches.
      mlmd_connection_config: ML metadata connection config.
      pipeline_info: The information of the pipeline that this node runs in.
      pipeline_runtime_spec: The runtime information of the pipeline that this
        node runs in.
      executor_spec: Specification for the executor of the node. This is
        expected for all nodes. This will be used to determine the
        specific ExecutorOperator class to be used to execute and will be passed
        into ExecutorOperator.
      custom_driver_spec: Specification for custom driver. This is expected only
        for advanced use cases.
      deployment_config: Deployment Config for the pipeline.
    """
    self._pipeline_node = pipeline_node
    self._mlmd_connection_config = mlmd_connection_config
    self._pipeline_info = pipeline_info
    self._pipeline_runtime_spec = pipeline_runtime_spec
    self._executor_spec = executor_spec
    self._custom_driver_spec = custom_driver_spec
    self._node_id = pipeline_node.node_info.id
    self._deployment_config = deployment_config

  def process(self, element: Any, *signals: Iterable[Any]) -> None:
    """Executes node based on signals.

    Args:
      element: a signal element to trigger the node.
      *signals: side input signals indicate completeness of upstream nodes.
    """
    for signal in signals:
      assert not list(signal), 'Signal PCollection should be empty.'

    logging.info('node %s is running.', self._node_id)
    self._run_component()
    logging.info('node %s is finished.', self._node_id)

  # TODO(b/171565775): rename this method to _run_node for this class and its
  # Childeren.
  def _run_component(self) -> None:
    launcher.Launcher(
        pipeline_node=self._pipeline_node,
        mlmd_connection=metadata.Metadata(self._mlmd_connection_config),
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._executor_spec,
        custom_driver_spec=self._custom_driver_spec).launch()


class BeamDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Beam."""
  _PIPELINE_NODE_DO_FN_CLS = PipelineNodeAsDoFn

  def __init__(self):
    """Initializes BeamDagRunner as a TFX orchestrator.
    """

  def _build_executable_spec(
      self, node_id: str,
      spec: any_pb2.Any) -> local_deployment_config_pb2.ExecutableSpec:
    """Builds ExecutableSpec given the any proto from IntermediateDeploymentConfig."""
    result = local_deployment_config_pb2.ExecutableSpec()
    if spec.Is(result.python_class_executable_spec.DESCRIPTOR):
      spec.Unpack(result.python_class_executable_spec)
    else:
      raise ValueError(
          'executor spec of {} is expected to be of one of the '
          'types of tfx.orchestration.deployment_config.ExecutableSpec.spec '
          'but got type {}'.format(node_id, spec.type_url))
    return result

  def _to_local_deployment(
      self,
      input_config: pipeline_pb2.IntermediateDeploymentConfig
  ) -> local_deployment_config_pb2.LocalDeploymentConfig:
    """Turns IntermediateDeploymentConfig to LocalDeploymentConfig."""
    result = local_deployment_config_pb2.LocalDeploymentConfig()
    for k, v in input_config.executor_specs.items():
      result.executor_specs[k].CopyFrom(self._build_executable_spec(k, v))

    for k, v in input_config.custom_driver_specs.items():
      result.custom_driver_specs[k].CopyFrom(self._build_executable_spec(k, v))

    if not input_config.metadata_connection_config.Unpack(
        result.metadata_connection_config):
      raise ValueError('metadata_connection_config is expected to be in type '
                       'ml_metadata.ConnectionConfig, but got type {}'.format(
                           input_config.metadata_connection_config.type_url))
    return result

  def _extract_deployment_config(
      self,
      pipeline: pipeline_pb2.Pipeline
  ) -> local_deployment_config_pb2.LocalDeploymentConfig:
    """Extracts the proto.Any pipeline.deployment_config to LocalDeploymentConfig."""

    if not pipeline.deployment_config:
      raise ValueError('deployment_config is not available in the pipeline.')

    result = local_deployment_config_pb2.LocalDeploymentConfig()
    if pipeline.deployment_config.Unpack(result):
      return result

    result = pipeline_pb2.IntermediateDeploymentConfig()
    if pipeline.deployment_config.Unpack(result):
      return self._to_local_deployment(result)

    raise ValueError("deployment_config's type {} is not supported".format(
        type(pipeline.deployment_config)))

  def _extract_executor_spec(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      node_id: str
  ) -> Optional[message.Message]:
    return self._unwrap_executable_spec(
        deployment_config.executor_specs.get(node_id))

  def _extract_custom_driver_spec(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      node_id: str
  ) -> Optional[message.Message]:
    return self._unwrap_executable_spec(
        deployment_config.custom_driver_specs.get(node_id))

  def _unwrap_executable_spec(
      self,
      executable_spec: Optional[local_deployment_config_pb2.ExecutableSpec]
  ) -> Optional[message.Message]:
    """Unwraps the one of spec from ExecutableSpec."""
    return (getattr(executable_spec, executable_spec.WhichOneof('spec'))
            if executable_spec else None)

  def _connection_config_from_deployment_config(self,
                                                deployment_config: Any) -> Any:
    return deployment_config.metadata_connection_config

  def run(self, pipeline: Union[pipeline_pb2.Pipeline,
                                pipeline_py.Pipeline]) -> None:
    """Deploys given logical pipeline on Beam.

    Args:
      pipeline: Logical pipeline in IR format.
    """
    # For CLI, while creating or updating pipeline, pipeline_args are extracted
    # and hence we avoid deploying the pipeline.
    if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
      return

    if isinstance(pipeline, pipeline_py.Pipeline):
      c = compiler.Compiler()
      pipeline = c.compile(pipeline)

    run_id = datetime.datetime.now().isoformat()
    # Substitute the runtime parameter to be a concrete run_id
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id,
        })

    deployment_config = self._extract_deployment_config(pipeline)
    connection_config = self._connection_config_from_deployment_config(
        deployment_config)

    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_RUNNER: 'beam'}):
      with beam.Pipeline() as p:
        # Uses for triggering the node DoFns.
        root = p | 'CreateRoot' >> beam.Create([None])

        # Stores mapping of node to its signal.
        signal_map = {}
        # pipeline.nodes are in topological order.
        for node in pipeline.nodes:
          # TODO(b/160882349): Support subpipeline
          pipeline_node = node.pipeline_node
          node_id = pipeline_node.node_info.id
          executor_spec = self._extract_executor_spec(
              deployment_config, node_id)
          custom_driver_spec = self._extract_custom_driver_spec(
              deployment_config, node_id)

          # Signals from upstream nodes.
          signals_to_wait = []
          for upstream_node in pipeline_node.upstream_nodes:
            assert upstream_node in signal_map, ('Nodes are not in '
                                                 'topological order')
            signals_to_wait.append(signal_map[upstream_node])
          logging.info('Node %s depends on %s.', node_id,
                       [s.producer.full_label for s in signals_to_wait])

          # Each signal is an empty PCollection. AsIter ensures a node will
          # be triggered after upstream nodes are finished.
          signal_map[node_id] = (
              root
              | 'Run[%s]' % node_id >> beam.ParDo(
                  self._PIPELINE_NODE_DO_FN_CLS(
                      pipeline_node=pipeline_node,
                      mlmd_connection_config=connection_config,
                      pipeline_info=pipeline.pipeline_info,
                      pipeline_runtime_spec=pipeline.runtime_spec,
                      executor_spec=executor_spec,
                      custom_driver_spec=custom_driver_spec,
                      deployment_config=deployment_config),
                  *[beam.pvalue.AsIter(s) for s in signals_to_wait]))
          logging.info('Node %s is scheduled.', node_id)
