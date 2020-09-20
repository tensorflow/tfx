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

import os
from typing import Any, Iterable, Optional

from absl import logging
import apache_beam as beam
from tfx.orchestration import metadata
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import tfx_runner
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import telemetry_utils

from google.protobuf import message


# TODO(jyzhao): confirm it's re-executable, add test case.
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(Any)
class _PipelineNodeAsDoFn(beam.DoFn):
  """Wrap component as beam DoFn."""

  def __init__(self,
               pipeline_node: pipeline_pb2.PipelineNode,
               mlmd_connection: metadata.Metadata,
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
               executor_spec: Optional[message.Message],
               custom_driver_spec: Optional[message.Message]):
    """Initializes the _PipelineNodeAsDoFn.

    Args:
      pipeline_node: The specification of the node that this launcher lauches.
      mlmd_connection: ML metadata connection. The connection is expected to
        not be opened before launcher is initiated.
      pipeline_info: The information of the pipeline that this node runs in.
      pipeline_runtime_spec: The runtime information of the pipeline that this
        node runs in.
      executor_spec: Specification for the executor of the node. This is
        expected for all components nodes. This will be used to determine the
        specific ExecutorOperator class to be used to execute and will be passed
        into ExecutorOperator.
      custom_driver_spec: Specification for custom driver. This is expected only
        for advanced use cases.
    """
    self._launcher = launcher.Launcher(
        pipeline_node=pipeline_node,
        mlmd_connection=mlmd_connection,
        pipeline_info=pipeline_info,
        pipeline_runtime_spec=pipeline_runtime_spec,
        executor_spec=executor_spec,
        custom_driver_spec=custom_driver_spec)
    self._component_id = pipeline_node.node_info.id

  def process(self, element: Any, *signals: Iterable[Any]) -> None:
    """Executes component based on signals.

    Args:
      element: a signal element to trigger the component.
      *signals: side input signals indicate completeness of upstream components.
    """
    for signal in signals:
      assert not list(signal), 'Signal PCollection should be empty.'
    self._run_component()

  def _run_component(self) -> None:
    logging.info('Component %s is running.', self._component_id)
    self._launcher.launch()
    logging.info('Component %s is finished.', self._component_id)


class BeamDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Beam."""

  def __init__(self):
    """Initializes BeamDagRunner as a TFX orchestrator.
    """

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

    raise ValueError("deployment_config's type {} is not supported".format(
        type(pipeline.deployment_config)))

  def _extract_executor_spec(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      component_id: str
  ) -> Optional[message.Message]:
    return self._unwrap_executable_spec(
        deployment_config.executor_specs.get(component_id))

  def _extract_custom_driver_spec(
      self,
      deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
      component_id: str
  ) -> Optional[message.Message]:
    return self._unwrap_executable_spec(
        deployment_config.custom_driver_specs.get(component_id))

  def _unwrap_executable_spec(
      self,
      executable_spec: Optional[local_deployment_config_pb2.ExecutableSpec]
  ) -> Optional[message.Message]:
    """Unwraps the one of spec from ExecutableSpec."""
    return (getattr(executable_spec, executable_spec.WhichOneof('spec'))
            if executable_spec else None)

  def run(self, pipeline: pipeline_pb2.Pipeline) -> None:
    """Deploys given logical pipeline on Beam.

    Args:
      pipeline: Logical pipeline in IR format.
    """
    # For CLI, while creating or updating pipeline, pipeline_args are extracted
    # and hence we avoid deploying the pipeline.
    if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
      return

    # TODO(b/163003901): Support beam DAG runner args through IR.
    deployment_config = self._extract_deployment_config(pipeline)
    connection_config = deployment_config.metadata_connection_config
    mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_RUNNER: 'beam'}):
      with beam.Pipeline() as p:
        # Uses for triggering the component DoFns.
        root = p | 'CreateRoot' >> beam.Create([None])

        # Stores mapping of component to its signal.
        signal_map = {}
        # pipeline.components are in topological order.
        for node in pipeline.nodes:
          # TODO(b/160882349): Support subpipeline
          pipeline_node = node.pipeline_node
          component_id = pipeline_node.node_info.id
          executor_spec = self._extract_executor_spec(
              deployment_config, component_id)
          custom_driver_spec = self._extract_custom_driver_spec(
              deployment_config, component_id)

          # Signals from upstream components.
          signals_to_wait = []
          for upstream_node in pipeline_node.upstream_nodes:
            assert upstream_node in signal_map, ('Components is not in '
                                                 'topological order')
            signals_to_wait.append(signal_map[upstream_node])
          logging.info('Component %s depends on %s.', component_id,
                       [s.producer.full_label for s in signals_to_wait])

          # Each signal is an empty PCollection. AsIter ensures component will
          # be triggered after upstream components are finished.
          # LINT.IfChange
          signal_map[component_id] = (
              root
              | 'Run[%s]' % component_id >> beam.ParDo(
                  _PipelineNodeAsDoFn(
                      pipeline_node=pipeline_node,
                      mlmd_connection=mlmd_connection,
                      pipeline_info=pipeline.pipeline_info,
                      pipeline_runtime_spec=pipeline.runtime_spec,
                      executor_spec=executor_spec,
                      custom_driver_spec=custom_driver_spec), *
                  [beam.pvalue.AsIter(s) for s in signals_to_wait]))
          # LINT.ThenChange(../beam/beam_dag_runner.py)
          logging.info('Component %s is scheduled.', component_id)
