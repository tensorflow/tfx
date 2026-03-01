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
"""Definition of Beam TFX runner."""

import datetime
from typing import Optional

from absl import logging
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.local import runner_utils
from tfx.orchestration.portable import launcher
from tfx.orchestration.portable import partial_run_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable import tfx_runner
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import doc_controls
from tfx.utils import telemetry_utils


class LocalDagRunner(tfx_runner.IrBasedRunner):
  """Local TFX DAG runner."""

  def __init__(self):
    """Initializes LocalDagRunner as a TFX orchestrator."""
    pass

  @doc_controls.do_not_generate_docs
  def run_with_ir(
      self,
      pipeline: pipeline_pb2.Pipeline,
      run_options: Optional[pipeline_pb2.RunOptions] = None,
  ) -> None:
    """Runs given pipeline locally.

    Args:
      pipeline: Pipeline IR containing pipeline args and components.
      run_options: Optional args for the run.

    Raises:
      ValueError: If run_options is provided, and partial_run_options.from_nodes
        and partial_run_options.to_nodes are both empty.
    """
    # Substitute the runtime parameter to be a concrete run_id
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_RUN_ID_PARAMETER_NAME:
                datetime.datetime.now().isoformat(),
        })

    deployment_config = runner_utils.extract_local_deployment_config(pipeline)
    connection_config = getattr(
        deployment_config.metadata_connection_config,
        deployment_config.metadata_connection_config.WhichOneof(
            'connection_config'))

    logging.info('Using deployment config:\n %s', deployment_config)
    logging.info('Using connection config:\n %s', connection_config)

    if run_options:
      logging.info('Using run_options:\n %s', run_options)
      pr_opts = run_options.partial_run
      partial_run_utils.mark_pipeline(
          pipeline,
          from_nodes=pr_opts.from_nodes or None,
          to_nodes=pr_opts.to_nodes or None,
          snapshot_settings=pr_opts.snapshot_settings)

    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_RUNNER: 'local'}):
      # Run each component. Note that the pipeline.components list is in
      # topological order.
      #
      # TODO(b/171319478): After IR-based execution is used, used multi-threaded
      # execution so that independent components can be run in parallel.
      for node in pipeline.nodes:
        pipeline_node = node.pipeline_node
        node_id = pipeline_node.node_info.id
        if pipeline_node.execution_options.HasField('skip'):
          logging.info('Skipping component %s.', node_id)
          continue
        executor_spec = runner_utils.extract_executor_spec(
            deployment_config, node_id)
        custom_driver_spec = runner_utils.extract_custom_driver_spec(
            deployment_config, node_id)

        component_launcher = launcher.Launcher(
            pipeline_node=pipeline_node,
            mlmd_connection=metadata.Metadata(connection_config),
            pipeline_info=pipeline.pipeline_info,
            pipeline_runtime_spec=pipeline.runtime_spec,
            executor_spec=executor_spec,
            custom_driver_spec=custom_driver_spec)
        logging.info('Component %s is running.', node_id)
        if pipeline_node.execution_options.run.perform_snapshot:
          with metadata.Metadata(connection_config) as mlmd_handle:
            partial_run_utils.snapshot(mlmd_handle, pipeline)
        component_launcher.launch()
        logging.info('Component %s is finished.', node_id)
