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
"""Tests for tfx.orchestration.experimental.interactive.interactive_context."""

import builtins
import os
import shutil
import tempfile
import textwrap
from typing import Any, Dict, List
from unittest import mock

import jinja2
import nbformat
import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration.experimental.interactive import interactive_context
from tfx.orchestration.experimental.interactive import standard_visualizations
from tfx.orchestration.launcher.in_process_component_launcher import InProcessComponentLauncher
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.utils import telemetry_utils


class InteractiveContextTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    builtins.__dict__['__IPYTHON__'] = True
    self._tmpdir = None

  def tearDown(self):
    if self._tmpdir:
      shutil.rmtree(self._tmpdir, ignore_errors=True)
    super().tearDown()

  def _setupTestNotebook(self, notebook_name='test_notebook.ipynb'):
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(source='A markdown cell.'),
            nbformat.v4.new_code_cell(source='foo = 1'),
            nbformat.v4.new_markdown_cell(source='Another markdown cell.'),
            nbformat.v4.new_code_cell(source=textwrap.dedent('''\
                %%skip_for_export
                !pip install something
                !ls
                x = 1
                y = 2
                print('this cell should not be exported')''')),
            nbformat.v4.new_code_cell(source=textwrap.dedent('''\
                def bar():
                  %some_line_magic print('this line should not be exported')
                  a = "hello"
                  b = "world"
                  return a + b''')),
            nbformat.v4.new_code_cell(source=textwrap.dedent('''\
                def baz():
                  c = "nyan"
                  d = "cat"
                  return c + d''')),
        ]
    )
    self._tmpdir = tempfile.mkdtemp()
    self._exportdir = tempfile.mkdtemp()
    self._notebook_fp = os.path.join(self._tmpdir, notebook_name)
    nbformat.write(notebook, self._notebook_fp)

  def testBasicRun(self):

    class _FakeComponentSpec(types.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):
      CALLED = False

      def Do(self, input_dict: Dict[str, List[types.Artifact]],
             output_dict: Dict[str, List[types.Artifact]],
             exec_properties: Dict[str, Any]) -> None:
        _FakeExecutor.CALLED = True

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)

      def __init__(self, spec: types.ComponentSpec):
        super().__init__(spec=spec)

    c = interactive_context.InteractiveContext()
    component = _FakeComponent(_FakeComponentSpec())
    c.run(component)
    self.assertTrue(_FakeExecutor.CALLED)

  def testRunMethodRequiresIPython(self):
    del builtins.__dict__['__IPYTHON__']

    c = interactive_context.InteractiveContext()
    self.assertIsNone(c.run(None))

  def testUnresolvedChannel(self):

    class _FakeComponentSpec(types.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {
          'input':
              component_spec.ChannelParameter(type=standard_artifacts.Examples)
      }
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):
      CALLED = False

      def Do(self, input_dict: Dict[str, List[types.Artifact]],
             output_dict: Dict[str, List[types.Artifact]],
             exec_properties: Dict[str, Any]) -> None:
        _FakeExecutor.CALLED = True

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)

      def __init__(self, spec: types.ComponentSpec):
        super().__init__(spec=spec)

    c = interactive_context.InteractiveContext()
    foo = types.Channel(type=standard_artifacts.Examples).set_artifacts(
        [standard_artifacts.Examples()])
    component = _FakeComponent(_FakeComponentSpec(input=foo))
    with self.assertRaisesRegex(ValueError, 'Unresolved input channel'):
      c.run(component)

  @mock.patch.object(jinja2.Environment, 'get_template',
                     return_value=jinja2.Template('{{ notebook_content }}'))
  def testExportToPipeline(self, mock_get_template):
    self._setupTestNotebook()

    c = interactive_context.InteractiveContext()
    export_filepath = os.path.join(self._exportdir, 'exported_pipeline.py')
    c.export_to_pipeline(notebook_filepath=self._notebook_fp,
                         export_filepath=export_filepath,
                         runner_type='beam')

    with open(export_filepath, 'r') as exported_pipeline:
      code = exported_pipeline.read()
      self.assertEqual(code, textwrap.dedent('''\
          foo = 1

          def bar():
            a = "hello"
            b = "world"
            return a + b

          def baz():
            c = "nyan"
            d = "cat"
            return c + d'''))

  def testExportToPipelineRaisesErrorInvalidRunnerType(self):
    self._setupTestNotebook()

    c = interactive_context.InteractiveContext()
    export_filepath = os.path.join(self._exportdir, 'exported_pipeline.py')
    with self.assertRaisesRegex(ValueError, 'runner_type'):
      c.export_to_pipeline(notebook_filepath=self._notebook_fp,
                           export_filepath=export_filepath,
                           runner_type='foobar')

  @mock.patch('tfx.orchestration.experimental.interactive.'
              'standard_visualizations.ExampleAnomaliesVisualization.display')
  def testShow(self, *unused_mocks):
    context = interactive_context.InteractiveContext()
    mock_object = mock.MagicMock()
    standard_visualizations.ExampleAnomaliesVisualization.display = mock_object
    mock_object.assert_not_called()
    artifact = standard_artifacts.ExampleAnomalies()
    context.show(
        types.Channel(type=standard_artifacts.ExampleAnomalies).set_artifacts(
            [artifact]))
    mock_object.assert_called_with(artifact)

  @mock.patch('tfx.orchestration.launcher.in_process_component_launcher.'
              'InProcessComponentLauncher.create')
  def testTelemetry(self, mock_launcher_create):

    class _FakeLauncher:

      def __init__(self):
        self.recorded_labels = []

      def launch(self):
        self.recorded_labels = telemetry_utils.make_beam_labels_args()
        return mock.MagicMock()

    class _FakeComponentSpec(types.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):

      def Do(self, input_dict: Dict[str, List[types.Artifact]],
             output_dict: Dict[str, List[types.Artifact]],
             exec_properties: Dict[str, Any]) -> None:
        pass

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)

      def __init__(self):
        super().__init__(spec=_FakeComponentSpec())

    # Set up fake on launcher.
    fake_launcher = _FakeLauncher()
    mock_launcher_create.side_effect = [fake_launcher]
    InProcessComponentLauncher.create = mock_launcher_create

    context = interactive_context.InteractiveContext()
    context.run(_FakeComponent())
    self.assertIn('--labels tfx_runner=interactivecontext',
                  ' '.join(fake_launcher.recorded_labels))


if __name__ == '__main__':
  tf.test.main()
