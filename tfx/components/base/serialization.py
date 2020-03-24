# Lint as: python2, python3
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
"""Component executor function serialization logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import inspect
import textwrap
import types
from typing import Text


class SourceCopySerializer(object):
  """Function serializer by source code extraction."""

  @classmethod
  def encode(cls, func: types.FunctionType) -> Text:
    """Encode the given function object for execution in Python entrypoint."""
    func_code = inspect.getsource(func)
    return cls._strip_typehints(textwrap.dedent(func_code))

  @classmethod
  def _strip_typehints(cls, source_code: Text) -> Text:
    """Strips type annotations from function definition text."""
    # Disable Pytype attribute error while traversing AST nodes.
    # pytype: disable=attribute-error

    # First, we parse the function definition text and identify the FunctionDef
    # node.
    parsed_module = ast.parse(source_code)
    if len(parsed_module.body) != 1:
      raise ValueError(
          'Could not parse function definition: multiple `FunctionDef`s in '
          'parsed module.')
    function_def = parsed_module.body[0]

    # Next, we get the line_number of the first statement in the function.
    first_statement_line_number = function_def.body[0].lineno

    # Now, we reconstruct the function by creating a "def" line, followed by the
    # lines of the function body, starting with the first statement.
    reconstructed_lines = []
    def_line = 'def %s(%s):' % (function_def.name, ', '.join(
        arg.arg for arg in function_def.args.args))
    # pytype: enable=attribute-error
    reconstructed_lines.append(def_line)
    source_lines = source_code.split('\n')
    reconstructed_lines += source_lines[first_statement_line_number - 1:]
    return '\n'.join(reconstructed_lines)
