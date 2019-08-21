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

Note: these APIs are **experimental** and major changes to interface and
functionality are expected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import functools
import inspect
import logging
import os
import tempfile

import nbformat
from six.moves import builtins
from typing import Text

from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import base_component
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.interactive import execution_result
from tfx.orchestration.interactive import notebook_formatters
from tfx.orchestration.launcher import in_proc_component_launcher


def requires_ipython(fn):
  """Decorator for methods that can only be run in IPython."""
  @functools.wraps(fn)
  def check_ipython(*args, **kwargs):
    """Invokes `fn` if called from IPython, otherwise just emits a warning."""
    # __IPYTHON__ variable is set by IPython, see
    # https://ipython.org/ipython-doc/rel-0.10.2/html/interactive/reference.html#embedding-ipython.
    if getattr(builtins, '__IPYTHON__', None):
      return fn(*args, **kwargs)
    else:
      logging.warning('Method "%s" is a no-op when invoked outside of IPython.',
                      fn.__name__)

  return check_ipython


class InteractiveContext(object):
  """TFX interactive context for interactive TFX notebook development.

  Note: these APIs are **experimental** and major changes to interface and
  functionality are expected.
  """

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
      pipeline_name = ('interactive-%s' %
                       datetime.datetime.now().isoformat().replace(':', '_'))
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

    # Register IPython formatters.
    notebook_formatters.register_formatters()

  @requires_ipython
  def run(self,
          component: base_component.BaseComponent,
          enable_cache: bool = True) -> execution_result.ExecutionResult:
    """Run a given TFX component in the interactive context.

    Args:
      component: Component instance to be run.
      enable_cache: whether caching logic should be enabled in the driver.

    Returns:
      execution_result.ExecutionResult object.
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
        artifact.producer_component = component.component_id
        artifact.run_id = run_id
        artifact.name = name
    # TODO(hongyes): figure out how to resolve launcher class in the interactive
    # context.
    launcher = in_proc_component_launcher.InProcComponentLauncher.create(
        component, pipeline_info, driver_args, self.metadata_connection_config,
        additional_pipeline_args)
    execution_id = launcher.launch()

    return execution_result.ExecutionResult(
        component=component,
        execution_id=execution_id)

  @requires_ipython
  def export_to_pipeline(self,
                         notebook_filename: Text,
                         pipeline_filename: Text = None):
    """Exports a notebook to a .py file as a runnable pipeline.

    The pipeline will be exported to the same directory as the notebook.

    Args:
      notebook_filename: String name of the notebook file, e.g.
        'notebook.ipynb'.
      pipeline_filename: String name for the exported pipeline python file, e.g.
        'exported_pipeline.py'. If `None`, a filename will be generated
        using `notebook_filename`.
    """
    current_frame = inspect.currentframe()
    if current_frame is None:
      raise ValueError('Unable to get current frame.')

    caller_filepath = inspect.getfile(current_frame.f_back)
    notebook_dir = os.path.dirname(os.path.abspath(caller_filepath))

    # The notebook filename is user-provided, as IPython kernels are agnostic to
    # notebook metadata by design, and it seems that existing workarounds to
    # retrieve the notebook filename are not universally robust
    # (https://github.com/jupyter/notebook/issues/1000).
    notebook_fp = os.path.join(notebook_dir, notebook_filename)

    if pipeline_filename is None:
      pipeline_filename = os.path.splitext(notebook_filename)[0] + '_export.py'
    pipeline_fp = os.path.join(notebook_dir, pipeline_filename)
    logging.info('Exporting contents of %s to %s.', notebook_fp, pipeline_fp)

    with open(notebook_fp) as notebook_f, open(pipeline_fp, 'w') as pipeline_f:
      notebook = nbformat.read(notebook_f, nbformat.NO_CONVERT)
      cells = notebook['cells']
      code_cells = (cell for cell in cells if cell['cell_type'] == 'code')
      pipeline_f.write(
          '\n\n'.join(code_cell['source'] for code_cell in code_cells))
