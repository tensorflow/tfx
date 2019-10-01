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
"""Tests for tfx.orchestration.interactive.interactive_context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import textwrap

import nbformat
from six.moves import builtins
import tensorflow as tf
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.orchestration.interactive import interactive_context
from tfx.types import component_spec


class InteractiveContextTest(tf.test.TestCase):

  def setUp(self):
    super(InteractiveContextTest, self).setUp()

    builtins.__dict__['__IPYTHON__'] = True
    self._tmpdir = None

  def tearDown(self):
    if self._tmpdir:
      shutil.rmtree(self._tmpdir, ignore_errors=True)
    super(InteractiveContextTest, self).tearDown()

  def _setupTestNotebook(self, notebook_name='test_notebook.ipynb'):
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(source='A markdown cell.'),
            nbformat.v4.new_code_cell(source='foo = 1'),
            nbformat.v4.new_markdown_cell(source='Another markdown cell.'),
            nbformat.v4.new_code_cell(source=textwrap.dedent('''\
                %%skip_for_export
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

  def testRequiresIPythonExecutes(self):
    self.foo_called = False
    def foo():
      self.foo_called = True

    interactive_context.requires_ipython(foo)()
    self.assertTrue(self.foo_called)

  def testRequiresIPythonNoOp(self):
    del builtins.__dict__['__IPYTHON__']

    self.foo_called = False
    def foo():
      self.foo_called = True
    interactive_context.requires_ipython(foo)()
    self.assertFalse(self.foo_called)

  def testBasicRun(self):

    class _FakeComponentSpec(types.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):
      CALLED = False

      def Do(self, input_dict: Dict[Text, List[types.Artifact]],
             output_dict: Dict[Text, List[types.Artifact]],
             exec_properties: Dict[Text, Any]) -> None:
        _FakeExecutor.CALLED = True

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)

      def __init__(self, spec: types.ComponentSpec):
        super(_FakeComponent, self).__init__(spec=spec)

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
      INPUTS = {'input': component_spec.ChannelParameter(type_name='Foo')}
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):
      CALLED = False

      def Do(self, input_dict: Dict[Text, List[types.Artifact]],
             output_dict: Dict[Text, List[types.Artifact]],
             exec_properties: Dict[Text, Any]) -> None:
        _FakeExecutor.CALLED = True

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(_FakeExecutor)

      def __init__(self, spec: types.ComponentSpec):
        super(_FakeComponent, self).__init__(spec=spec)

    c = interactive_context.InteractiveContext()
    foo = types.Channel(type_name='Foo', artifacts=[types.Artifact('Foo')])
    component = _FakeComponent(_FakeComponentSpec(input=foo))
    with self.assertRaisesRegexp(ValueError, 'Unresolved input channel'):
      c.run(component)

  def testExportToPipeline(self):
    self._setupTestNotebook()

    c = interactive_context.InteractiveContext()
    export_filepath = os.path.join(self._exportdir, 'exported_pipeline.py')
    c.export_to_pipeline(notebook_filepath=self._notebook_fp,
                         export_filepath=export_filepath)

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


if __name__ == '__main__':
  tf.test.main()
