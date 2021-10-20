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
from typing import Any, Iterable, List, Optional

from absl import logging
import apache_beam as beam
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.beam.legacy import beam_dag_runner as legacy_beam_dag_runner
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.local import runner_utils
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import partial_run_utils
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
               deployment_config: Optional[message.Message],
               pipeline: Optional[pipeline_pb2.Pipeline]):
    """Initializes the PipelineNodeAsDoFn.

    Args:
      pipeline_node: The specification of the node that this launcher lauches.
      mlmd_connection_config: ML metadata connection config.
      pipeline_info: The information of the pipeline that this node runs in.
      pipeline_runtime_spec: The runtime information of the pipeline that this
        node runs in.
      executor_spec: Specification for the executor of the node. This is
        expected for all nodes. This will be used to determine the specific
        ExecutorOperator class to be used to execute and will be passed into
        ExecutorOperator.
      custom_driver_spec: Specification for custom driver. This is expected only
        for advanced use cases.
      deployment_config: Deployment Config for the pipeline.
      pipeline: Optional for full run, required for partial run.
    """
    self._pipeline_node = pipeline_node
    self._mlmd_connection_config = mlmd_connection_config
    self._pipeline_info = pipeline_info
    self._pipeline_runtime_spec = pipeline_runtime_spec
    self._executor_spec = executor_spec
    self._custom_driver_spec = custom_driver_spec
    self._node_id = pipeline_node.node_info.id
    self._deployment_config = deployment_config
    self._pipeline = pipeline

  def process(self, element: Any, *signals: Iterable[Any]) -> None:
    """Executes node based on signals.

    Args:
      element: a signal element to trigger the node.
      *signals: side input signals indicate completeness of upstream nodes.
    """
    for signal in signals:
      assert not list(signal), 'Signal PCollection should be empty.'

    logging.info('node %s is running.', self._node_id)
    self._run_node()
    logging.info('node %s is finished.', self._node_id)

  def _run_node(self) -> None:
    platform_config = self._extract_platform_config(self._deployment_config,
                                                    self._node_id)
    if self._pipeline_node.execution_options.run.perform_snapshot:
      with metadata.Metadata(self._mlmd_connection_config) as mlmd_handle:
        partial_run_utils.snapshot(mlmd_handle, self._pipeline)
    launcher.Launcher(
        pipeline_node=self._pipeline_node,
        mlmd_connection=metadata.Metadata(self._mlmd_connection_config),
        pipeline_info=self._pipeline_info,
        pipeline_runtime_spec=self._pipeline_runtime_spec,
        executor_spec=self._executor_spec,
        platform_config=platform_config,
        custom_driver_spec=self._custom_driver_spec).launch()

  def _extract_platform_config(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      node_id: str) -> Optional[message.Message]:
    platform_config = deployment_config.node_level_platform_configs.get(node_id)
    return (getattr(platform_config, platform_config.WhichOneof('config'))
            if platform_config else None)


class BeamDagRunner(tfx_runner.IrBasedRunner):
  """Tfx runner on Beam."""

  _PIPELINE_NODE_DO_FN_CLS = PipelineNodeAsDoFn

  def __new__(cls,
              beam_orchestrator_args: Optional[List[str]] = None,
              config: Optional[pipeline_config.PipelineConfig] = None):
    """Initializes BeamDagRunner as a TFX orchestrator.

    Create the legacy BeamDagRunner object if any of the legacy
    `beam_orchestrator_args` or `config` arguments are passed. A migration
    guide will be provided in a future TFX version for users of these arguments.

    Args:
      beam_orchestrator_args: Deprecated beam args for the beam orchestrator.
        Note that this is different from the beam_pipeline_args within
        additional_pipeline_args, which is for beam pipelines in components. If
        this option is used, the legacy non-IR-based BeamDagRunner will be
        constructed.
      config: Deprecated optional pipeline config for customizing the launching
        of each component. Defaults to pipeline config that supports
        InProcessComponentLauncher and DockerComponentLauncher. If this option
        is used, the legacy non-IR-based BeamDagRunner will be constructed.

    Returns:
      Legacy or IR-based BeamDagRunner object.
    """
    if beam_orchestrator_args or config:
      logging.info(
          'Using the legacy BeamDagRunner since `beam_orchestrator_args` or '
          '`config` argument was passed.')
      return legacy_beam_dag_runner.BeamDagRunner(
          beam_orchestrator_args=beam_orchestrator_args, config=config)
    else:
      return super(BeamDagRunner, cls).__new__(cls)

  def _extract_platform_config(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      node_id: str) -> Optional[message.Message]:
    platform_config = deployment_config.node_level_platform_configs.get(node_id)
    return (getattr(platform_config, platform_config.WhichOneof('config'))
            if platform_config else None)

  def _build_local_platform_config(
      self, node_id: str,
      spec: any_pb2.Any) -> local_deployment_config_pb2.LocalPlatformConfig:
    """Builds LocalPlatformConfig given the any proto from IntermediateDeploymentConfig."""
    result = local_deployment_config_pb2.LocalPlatformConfig()
    if spec.Is(result.docker_platform_config.DESCRIPTOR):
      spec.Unpack(result.docker_platform_config)
    else:
      raise ValueError(
          'Platform config of {} is expected to be of one of the '
          'types of tfx.orchestration.deployment_config.LocalPlatformConfig.config '
          'but got type {}'.format(node_id, spec.type_url))
    return result

  def _extract_deployment_config(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> local_deployment_config_pb2.LocalDeploymentConfig:
    """Extracts the proto.Any pipeline.deployment_config to LocalDeploymentConfig."""
    return runner_utils.extract_local_deployment_config(pipeline)

  def _extract_executor_spec(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      node_id: str) -> Optional[message.Message]:
    return runner_utils.extract_executor_spec(deployment_config, node_id)

  def _extract_custom_driver_spec(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      node_id: str) -> Optional[message.Message]:
    return runner_utils.extract_custom_driver_spec(deployment_config, node_id)

  def _connection_config_from_deployment_config(self,
                                                deployment_config: Any) -> Any:
    return deployment_config.metadata_connection_config

  def run_with_ir(self, pipeline: pipeline_pb2.Pipeline) -> None:
    """Deploys given logical pipeline on Beam.

    Args:
      pipeline: Logical pipeline in IR format.
    """
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S.%f')
    # Substitute the runtime parameter to be a concrete run_id
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id,
        })

    deployment_config = self._extract_deployment_config(pipeline)
    connection_config = self._connection_config_from_deployment_config(
        deployment_config)

    logging.info('Using deployment config:\n %s', deployment_config)
    logging.info('Using connection config:\n %s', connection_config)

    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_RUNNER: 'beam'}):
      with beam.Pipeline() as p:
        # Uses for triggering the node DoFns.
        root = p | 'CreateRoot' >> beam.Create([None])

        # Stores mapping of node_id to its signal.
        signal_map = {}
        snapshot_node_id = None
        # pipeline.nodes are in topological order.
        for node in pipeline.nodes:
          # TODO(b/160882349): Support subpipeline
          pipeline_node = node.pipeline_node
          node_id = pipeline_node.node_info.id
          if pipeline_node.execution_options.HasField('skip'):
            logging.info('Node %s is skipped in this partial run.', node_id)
            continue
          run_opts = pipeline_node.execution_options.run
          if run_opts.perform_snapshot:
            snapshot_node_id = node_id
          executor_spec = self._extract_executor_spec(deployment_config,
                                                      node_id)
          custom_driver_spec = self._extract_custom_driver_spec(
              deployment_config, node_id)

          # Signals from upstream nodes.
          signals_to_wait = []
          # This ensures that nodes that rely on artifacts reused from previous
          # pipelines runs are scheduled after the snapshot node.
          if run_opts.depends_on_snapshot and node_id != snapshot_node_id:
            signals_to_wait.append(signal_map[snapshot_node_id])
          for upstream_node_id in pipeline_node.upstream_nodes:
            if upstream_node_id in signal_map:
              signals_to_wait.append(signal_map[upstream_node_id])
            else:
              logging.info(
                  'Node %s is upstream of Node %s, but will be skipped in '
                  'this partial run. Node %s is responsible for ensuring that '
                  'Node %s\'s dependencies can be resolved.', upstream_node_id,
                  node_id, snapshot_node_id, node_id)
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
                      deployment_config=deployment_config,
                      pipeline=pipeline),
                  (*[beam.pvalue.AsIter(s) for s in signals_to_wait])))
          logging.info('Node %s is scheduled.', node_id)
