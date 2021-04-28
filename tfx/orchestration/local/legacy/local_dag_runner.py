# Lint as: python3
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
import os
from typing import Optional

from absl import logging

from tfx.dsl.components.base import base_component
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.config import config_utils
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.launcher import docker_component_launcher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.utils import telemetry_utils


class LocalDagRunner(tfx_runner.TfxRunner):
  """Local TFX DAG runner."""
  # TODO(b/171319478): We should use IR-based execution in this DAG runner.

  def __init__(self,
               config: Optional[pipeline_config.PipelineConfig] = None):
    """Initializes local TFX orchestrator.

    Args:
      config: Optional pipeline config for customizing the launching of each
        component. Defaults to pipeline config that supports
        InProcessComponentLauncher and DockerComponentLauncher.
    """
    if config is None:
      config = pipeline_config.PipelineConfig(
          supported_launcher_classes=[
              in_process_component_launcher.InProcessComponentLauncher,
              docker_component_launcher.DockerComponentLauncher,
          ],
      )
    super(LocalDagRunner, self).__init__(config)

  def run(self, tfx_pipeline: pipeline.Pipeline) -> None:
    """Runs given logical pipeline locally.

    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.
    """
    # For CLI, while creating or updating pipeline, pipeline_args are extracted
    # and hence we avoid executing the pipeline.
    if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
      return

    tfx_pipeline.pipeline_info.run_id = datetime.datetime.now().isoformat()

    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_RUNNER: 'local'}):
      # Run each component. Note that the pipeline.components list is in
      # topological order.
      #
      # TODO(b/171319478): After IR-based execution is used, used multi-threaded
      # execution so that independent components can be run in parallel.
      for component in tfx_pipeline.components:
        # TODO(b/187122662): Pass through pip dependencies as a first-class
        # component flag.
        if isinstance(component, base_component.BaseComponent):
          component._resolve_pip_dependencies(  # pylint: disable=protected-access
              tfx_pipeline.pipeline_info.pipeline_root)
        (component_launcher_class, component_config) = (
            config_utils.find_component_launch_info(self._config, component))
        driver_args = data_types.DriverArgs(
            enable_cache=tfx_pipeline.enable_cache)
        metadata_connection = metadata.Metadata(
            tfx_pipeline.metadata_connection_config)
        node_launcher = component_launcher_class.create(
            component=component,
            pipeline_info=tfx_pipeline.pipeline_info,
            driver_args=driver_args,
            metadata_connection=metadata_connection,
            beam_pipeline_args=tfx_pipeline.beam_pipeline_args,
            additional_pipeline_args=tfx_pipeline.additional_pipeline_args,
            component_config=component_config)
        logging.info('Component %s is running.', component.id)
        node_launcher.launch()
        logging.info('Component %s is finished.', component.id)
