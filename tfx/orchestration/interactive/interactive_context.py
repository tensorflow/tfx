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
"""TFX interactive context for iterative development.

See `examples/chicago_taxi_pipeline/taxi_pipeline_interactive.ipynb` for an
example of how to run TFX in a Jupyter notebook for iterative development.

Note: these APIs are experimental and changes to interface and functionality
are expected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
import tempfile
from typing import Text


from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import base_component
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.component_launcher import ComponentLauncher
from tfx.orchestration.component_launcher import ExecutionResult


class InteractiveContext(object):
  """TFX interactive context for interactive TFX notebook development."""

  _DEFAULT_SQLITE_FILENAME = 'metadata.sqlite'

  def __init__(
      self,
      pipeline_name: Text = None,
      pipeline_root: Text = None,
      metadata_connection_config: metadata_store_pb2.ConnectionConfig = None):
    """Initialize an InteractiveContext.

    Args:
      pipeline_name: Optional name of the pipeline for ML Metadata tracking
        purposes. If not specified, a name will be generated for you.
      pipeline_root: Optional path to the root of the pipeline's outputs. If not
        specified, an ephemeral temporary directory will be created and used.
      metadata_connection_config: Optional metadata_store_pb2.ConnectionConfig
        instance used to configure connection to a ML Metadata connection. If
        not specified, an ephemeral SQLite MLMD connection contained in the
        pipeline_root directory with file name "metadata.sqlite" will be used.
    """

    if not pipeline_name:
      pipeline_name = 'interactive-%s' % datetime.datetime.now().isoformat()
    if not pipeline_root:
      pipeline_root = tempfile.mkdtemp(prefix='tfx-%s-' % pipeline_name)
      logging.info(
          'InteractiveContext pipeline_root argument not provided: using '
          'temporary directory %s as root for pipeline outputs.',
          pipeline_root)
    if not metadata_connection_config:
      # TODO(ccy): consider reconciling similar logic here with other instances
      # in tfx/orchestration/...
      metadata_sqlite_path = os.path.join(
          pipeline_root, self._DEFAULT_SQLITE_FILENAME)
      metadata_connection_config = metadata.sqlite_metadata_connection_config(
          metadata_sqlite_path)
      logging.info(
          'InteractiveContext metadata_connection_config not provided: using '
          'SQLite ML Metadata database at %s.',
          metadata_sqlite_path)
    self.pipeline_name = pipeline_name
    self.pipeline_root = pipeline_root
    self.metadata_connection_config = metadata_connection_config

    # Register IPython formatters. Import this here to avoid circular
    # dependency.
    # pylint: disable=g-import-not-at-top
    try:
      from tfx.orchestration.interactive import notebook_formatters  # pytype: disable=import-error
      # pylint: enable=g-import-not-at-top
      notebook_formatters.register_formatters()
    except ImportError:
      pass

  def run(self,
          component: base_component.BaseComponent,
          enable_cache: bool = True) -> ExecutionResult:
    """Run a given TFX component in the interactive context.

    Args:
      component: Component instance to be run.
      enable_cache: whether caching logic should be enabled in the driver.

    Returns:
      ExecutionResult object.
    """
    run_id = datetime.datetime.now().isoformat()
    pipeline_info = data_types.PipelineInfo(
        pipeline_name=self.pipeline_name,
        pipeline_root=self.pipeline_root,
        run_id=run_id)
    driver_args = data_types.DriverArgs(
        enable_cache=enable_cache,
        interactive_resolution=True)
    additional_pipeline_args = {}
    for name, output in component.outputs.get_all().items():
      for artifact in output.get():
        artifact.pipeline_name = self.pipeline_name
        artifact.producer_component = component.component_name
        artifact.run_id = run_id
        artifact.name = name
    launcher = ComponentLauncher(component, pipeline_info, driver_args,
                                 self.metadata_connection_config,
                                 additional_pipeline_args)
    return launcher.launch()
