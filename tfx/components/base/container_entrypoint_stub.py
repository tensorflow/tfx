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
"""Self-contained entrypoint stub for Python raw container components.

Raw container components may not have TFX installed, so we provide a minimal
set of utilities and an entrypoint. This stub code (with user-specific
component executor code inserted) will be executed in the container using
`python -c <code>`.

The stub starts with the header line "# BEGIN_STUB" and ends with the footer
line "# END_STUB". Component-specific user code should be placed at the
"# INSERTION_POINT" marker.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-statement-before-imports
# These inputs will be inserted (at the INSERTION_POINT below) for the specific
# user component.
# String name of the component function.
_COMPONENT_NAME = 'Component'
# Reference to the component function.
_COMPONENT_FUNCTION = lambda *args: None
# List of tuples of (arg_name, arg_format).
_COMPONENT_ARGS = []
# Map from primitive value return key name to value type, where the value type
# is one of `int`, `float`, `str` or `bytes`.
_COMPONENT_RETURN_VALUE_TYPES = {}

# BEGIN_STUB
import argparse
import enum
import json
import logging
import struct
import sys
from typing import Any, Dict, Text, Type


class ArgFormats(enum.Enum):
  """Component executor function argument formats.

  Should be kept in sync with `tfx.components.base.function_parser.ArgFormat`.
  """
  INPUT_ARTIFACT = 1
  INPUT_ARTIFACT_URI = 2
  OUTPUT_ARTIFACT = 3
  OUTPUT_ARTIFACT_URI = 4
  ARTIFACT_VALUE = 5


class Artifact(object):
  """TFX Artifact compatibility stub.

  Provides the ability to use Artifact objects in a Python environment without
  TFX installed.
  """

  _TYPE_DEFAULTS = {'INT': 0, 'FLOAT': 0.0, 'STRING': ''}

  def __init__(self, type_name: Text, type_properties: Dict[Text, Text],
               uri: Text, properties: Dict[Text, Any]):
    self._type_name = type_name
    self._type_properties = type_properties
    self.uri = uri
    self._properties = properties

  def __getattr__(self, name: Text) -> Any:
    if name not in self._type_properties:
      raise AttributeError('Artifact type %r has no property %r.' %
                           (self._type_name, name))
    if name in self._properties:
      return self._properties[name]
    else:
      return Artifact._TYPE_DEFAULTS[self._type_properties[name]]

  @classmethod
  def from_json_dict(cls, data: Dict[Text, Any]) -> Any:
    """Deserialize Artifact from JSON dictionary."""
    type_name = data['artifact_type']['name']
    type_properties = data['artifact_type']['properties']
    uri = data['artifact']['uri']
    # TODO(ccy): support custom properties.
    properties = {}
    for name, value_struct in data['artifact']['properties'].items():
      if type_properties[name] == 'INT':
        properties[name] = int(value_struct['int_value'])
      elif type_properties[name] == 'FLOAT':
        properties[name] = float(value_struct['float_value'])
      elif type_properties[name] == 'STRING':
        properties[name] = value_struct['string_value']
      else:
        raise ValueError('Unknown artifact property type: %r.' %
                         (type_properties[name],))

    return Artifact(type_name, type_properties, uri, properties)


def deserialize_artifact(artifact_dict: Dict[Text, Any]) -> Any:
  """Deserialize the given JSON-encoded artifact dictionary."""
  try:
    # Use the TFX package if possible.
    from tfx import types  # pylint: disable=g-import-not-at-top, import-outside-toplevel # pytype: disable=import-error
    types.Artifact.from_json_dict(artifact_dict)
  except ImportError:
    # Fall back to the utility provided here.
    return Artifact.from_json_dict(artifact_dict)


def get_parser():
  """Construct argparse parser for component entrypoint."""
  parser = argparse.ArgumentParser(_COMPONENT_NAME)
  for arg_name, arg_format in _COMPONENT_ARGS:
    if arg_format == ArgFormats.INPUT_ARTIFACT:
      help_string = 'Artifact JSON for %r input artifact.' % arg_name
    elif arg_format == ArgFormats.OUTPUT_ARTIFACT:
      help_string = 'Artifact JSON for %r output artifact.' % arg_name
    elif arg_format == ArgFormats.INPUT_ARTIFACT_URI:
      help_string = 'Input artifact URI for %r.' % arg_name
    elif arg_format == ArgFormats.OUTPUT_ARTIFACT_URI:
      help_string = 'Output artifact URI for %r.' % arg_name
    elif arg_format == ArgFormats.ARTIFACT_VALUE:
      help_string = 'Input value for %r.' % arg_name
    else:
      raise ValueError('Invalid arg_format: %r.' % arg_format)
    parser.add_argument(
        '--%s' % arg_name, type=Text, required=True, help=help_string)
  for output_name in _COMPONENT_RETURN_VALUE_TYPES:
    parser.add_argument(
        '--%s' % output_name,
        type=Text,
        required=True,
        help='Output artifact URI for %r.')
  return parser


def _encode_value(value: Any, value_type: Type[Any]) -> bytes:
  """Encode value as serialized bytes."""
  # The serialization format here should match those of the ValueArtifacts
  # defined in `tfx.types.standard_artifacts`.
  if value_type == int:
    return struct.pack('>i', value)
  elif value_type == float:
    return struct.pack('>d', value)
  elif value_type == bytes:
    return value
  elif value_type == Text:
    return value.encode('utf-8')
  else:
    raise ValueError('Unknown value_type: %r.' % value_type)


def process_outputs(outputs: Dict[Text, Any], output_uris: Dict[Text, Text]):
  """Process outputs from running component."""
  # TODO(ccy): Support output artifact property modifications.
  if not _COMPONENT_RETURN_VALUE_TYPES:
    return

  try:
    import tensorflow as tf  # pylint: disable=g-import-not-at-top,g-explicit-tensorflow-version-import,import-outside-toplevel
  except ImportError:
    raise EnvironmentError(
        'Could not import tensorflow within container to write component '
        'outputs.')

  outputs = outputs or {}
  if not isinstance(outputs, dict):
    logging.warning(
        'Expected component executor function to return a dict of outputs.')
    return
  unprocessed_outputs = set(outputs.keys())
  for name, expected_type in _COMPONENT_RETURN_VALUE_TYPES.items():
    if name not in unprocessed_outputs:
      logging.warning(
          'Did not receive expected output %r as return value from '
          'component.', name)
      continue
    value = outputs[name]
    if not isinstance(value, expected_type):
      logging.warning(
          'Expected return value for output %r to be of type %s, but got %r '
          'instead', name, expected_type, value)
      continue
    encoded_value = _encode_value(value, expected_type)
    tf.io.gfile.GFile(output_uris[name], 'wb').write(encoded_value)


def run_component():
  """Python raw container component entrypoint."""
  parser = get_parser()
  args = parser.parse_args(sys.argv[1:])

  component_args = []
  for arg_name, arg_format in _COMPONENT_ARGS:
    value = json.loads(getattr(args, arg_name))
    if arg_format in (ArgFormats.INPUT_ARTIFACT, ArgFormats.OUTPUT_ARTIFACT):
      component_args.append(deserialize_artifact(value))
    elif arg_format in (ArgFormats.INPUT_ARTIFACT_URI,
                        ArgFormats.OUTPUT_ARTIFACT_URI):
      component_args.append(value)
    elif arg_format == ArgFormats.ARTIFACT_VALUE:
      component_args.append(value)
  output_uris = {}
  for output_name in _COMPONENT_RETURN_VALUE_TYPES:
    output_uris[output_name] = getattr(args, output_name)
  process_outputs(_COMPONENT_FUNCTION(*component_args), output_uris)  # pytype: disable=wrong-arg-types


# Begin user code.
# INSERTION_POINT
# End user code.

if __name__ == '__main__':
  run_component()

# END_STUB
