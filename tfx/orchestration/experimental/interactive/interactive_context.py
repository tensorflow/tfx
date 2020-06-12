# Lint as: python2, python3
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

import cgi
import datetime
import functools
import os
import tempfile
from typing import List, Optional, Text

import absl
import jinja2
import nbformat
from six.moves import builtins

from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.experimental.interactive import execution_result
from tfx.orchestration.experimental.interactive import notebook_formatters
from tfx.orchestration.experimental.interactive import standard_visualizations
from tfx.orchestration.experimental.interactive import visualizations
from tfx.orchestration.launcher import in_process_component_launcher

_SKIP_FOR_EXPORT_MAGIC = '%%skip_for_export'
_MAGIC_PREFIX = '%'
_CMD_LINE_PREFIX = '!'
_EXPORT_TEMPLATES_DIR = 'export_templates'


def requires_ipython(fn):
  """Decorator for methods that can only be run in IPython."""

  @functools.wraps(fn)
  def run_if_ipython(*args, **kwargs):
    """Invokes `fn` if called from IPython, otherwise just emits a warning."""
    if getattr(builtins, '__IPYTHON__', None):
      # __IPYTHON__ variable is set by IPython, see
      # https://ipython.org/ipython-doc/rel-0.10.2/html/interactive/reference.html#embedding-ipython.
      return fn(*args, **kwargs)
    else:
      absl.logging.warning(
          'Method "%s" is a no-op when invoked outside of IPython.',
          fn.__name__)

  return run_if_ipython


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
      absl.logging.warning(
          'InteractiveContext pipeline_root argument not provided: using '
          'temporary directory %s as root for pipeline outputs.', pipeline_root)
    if not metadata_connection_config:
      # TODO(ccy): consider reconciling similar logic here with other instances
      # in tfx/orchestration/...
      metadata_sqlite_path = os.path.join(pipeline_root,
                                          self._DEFAULT_SQLITE_FILENAME)
      metadata_connection_config = metadata.sqlite_metadata_connection_config(
          metadata_sqlite_path)
      absl.logging.warning(
          'InteractiveContext metadata_connection_config not provided: using '
          'SQLite ML Metadata database at %s.', metadata_sqlite_path)
    self.pipeline_name = pipeline_name
    self.pipeline_root = pipeline_root
    self.metadata_connection_config = metadata_connection_config

    # Register IPython formatters.
    notebook_formatters.register_formatters()

    # Register artifact visualizations.
    standard_visualizations.register_standard_visualizations()

  @requires_ipython
  def run(
      self,
      component: base_node.BaseNode,
      enable_cache: bool = True,
      beam_pipeline_args: Optional[List[Text]] = None
  ) -> execution_result.ExecutionResult:
    """Run a given TFX component in the interactive context.

    Args:
      component: Component instance to be run.
      enable_cache: whether caching logic should be enabled in the driver.
      beam_pipeline_args: Optional Beam pipeline args for beam jobs within
        executor. Executor will use beam DirectRunner as Default.

    Returns:
      execution_result.ExecutionResult object.
    """
    run_id = datetime.datetime.now().isoformat()
    pipeline_info = data_types.PipelineInfo(
        pipeline_name=self.pipeline_name,
        pipeline_root=self.pipeline_root,
        run_id=run_id)
    driver_args = data_types.DriverArgs(
        enable_cache=enable_cache, interactive_resolution=True)
    metadata_connection = metadata.Metadata(self.metadata_connection_config)
    beam_pipeline_args = beam_pipeline_args or []
    additional_pipeline_args = {}
    for name, output in component.outputs.items():
      for artifact in output.get():
        artifact.pipeline_name = self.pipeline_name
        artifact.producer_component = component.id
        artifact.name = name
    # TODO(hongyes): figure out how to resolve launcher class in the interactive
    # context.
    launcher = in_process_component_launcher.InProcessComponentLauncher.create(
        component, pipeline_info, driver_args, metadata_connection,
        beam_pipeline_args, additional_pipeline_args)
    execution_id = launcher.launch().execution_id

    return execution_result.ExecutionResult(
        component=component, execution_id=execution_id)

  @requires_ipython
  def export_to_pipeline(self, notebook_filepath: Text, export_filepath: Text,
                         runner_type: Text):
    """Exports a notebook to a .py file as a runnable pipeline.

    Args:
      notebook_filepath: String path of the notebook file, e.g.
        '/path/to/notebook.ipynb'.
      export_filepath: String path for the exported pipeline python file, e.g.
        '/path/to/exported_pipeline.py'.
      runner_type: String indicating type of runner, e.g. 'beam', 'airflow'.
    """
    if runner_type not in ['beam', 'airflow']:
      raise ValueError('Invalid runner_type: %s' % runner_type)

    absl.logging.info('Exporting contents of %s to %s with %s runner.',
                      notebook_filepath, export_filepath, runner_type)

    with open(notebook_filepath) as notebook_f,\
        open(export_filepath, 'w') as export_f:
      notebook = nbformat.read(notebook_f, nbformat.NO_CONVERT)
      cells = notebook['cells']
      code_cells = (cell for cell in cells if cell['cell_type'] == 'code')
      sources = []
      num_skipped_cells = 0
      for code_cell in code_cells:
        cell_source = code_cell['source']
        if cell_source.lstrip().startswith(_SKIP_FOR_EXPORT_MAGIC):
          num_skipped_cells += 1
          continue

        # Filter out all line/cell magics using `%` prefix and command line
        # invocations (e.g. !pip install ...).
        # Note: This will not work for magics invoked without the prefix when
        # %automagic is set.
        sources.append(('\n'.join(
            line for line in cell_source.split('\n')
            if not (line.lstrip().startswith(_MAGIC_PREFIX) or
                    line.lstrip().startswith(_CMD_LINE_PREFIX)))))

      jinja_env = jinja2.Environment(
          loader=jinja2.PackageLoader(
              __name__, package_path=_EXPORT_TEMPLATES_DIR))
      template_name = 'export_%s.tmpl' % runner_type
      # TODO(b/142326292): Consider parameterizing the other variables names
      # present in the export templates.
      rendered_template = jinja_env.get_template(template_name).render({
          'notebook_content': '\n\n'.join(sources),
      })
      export_f.write(rendered_template)
      absl.logging.info('%d cell(s) marked with "%s", skipped.',
                        num_skipped_cells, _SKIP_FOR_EXPORT_MAGIC)

  @requires_ipython
  def show(self, item: object) -> None:
    """Show the given object in an IPython notebook display."""
    from IPython.core.display import display  # pylint: disable=g-import-not-at-top
    from IPython.core.display import HTML  # pylint: disable=g-import-not-at-top
    if isinstance(item, types.Channel):
      channel = item
      artifacts = channel.get()
      for artifact in artifacts:
        artifact_heading = 'Artifact at %s' % cgi.escape(artifact.uri)  # pylint: disable=deprecated-method
        display(HTML('<b>%s</b><br/><br/>' % artifact_heading))
        visualization = visualizations.get_registry().get_visualization(
            artifact.type_name)
        if visualization:
          visualization.display(artifact)
    else:
      display(item)
