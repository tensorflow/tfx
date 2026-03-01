# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Utility for context manager testing."""

import textwrap

from absl.testing import absltest
from tfx.dsl.components.base import base_node
from tfx.dsl.context_managers import dsl_context
from tfx.dsl.context_managers import dsl_context_manager
from tfx.dsl.context_managers import dsl_context_registry


class _TestContext(dsl_context.DslContext):

  def __init__(self, name):
    self._name = name

  def __str__(self):
    return self._name


class TestContext(dsl_context_manager.DslContextManager[_TestContext]):

  def __init__(self, name):
    super().__init__()
    self._name = name

  def create_context(self):
    return _TestContext(self._name)

  def enter(self, context):
    return context


class Node(base_node.BaseNode):
  inputs = {}
  outputs = {}
  exec_properties = {}

  def __init__(self, name):
    super().__init__()
    self.with_id(name)


def _debug_str(reg: dsl_context_registry.DslContextRegistry):
  """Returns a debug string for the registry."""
  frame = []
  output_builder = []
  indent = 0
  for node in reg.all_nodes:
    next_frame = reg.get_contexts(node)
    for context in reversed(frame):
      if context not in next_frame:
        assert frame.pop() == context
        indent -= 2
        output_builder.append(' ' * indent + '}')
    assert frame == next_frame[: len(frame)]
    for context in next_frame:
      if context not in frame:
        if frame:
          assert context.parent == frame[-1]
        else:
          assert context.parent is None
        frame.append(context)
        output_builder.append(' ' * indent + f'{context} {{')
        indent += 2
    output_builder.append(' ' * indent + node.id)
  while frame:
    frame.pop()
    indent -= 2
    output_builder.append(' ' * indent + '}')
  return '\n'.join(output_builder)


def assert_registry_equal(
    test_case: absltest.TestCase,
    reg: dsl_context_registry.DslContextRegistry,
    expected_debug_str: str,
):
  expected = textwrap.dedent(expected_debug_str).strip()
  actual = _debug_str(reg)
  return test_case.assertEqual(
      expected, actual, f'Expected:\n{expected}\n\nActual:\n{actual}\n'
  )
