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
"""TFX RuntimeStringParameter placeholder.

Used to parameterize some properties of TFX components at runtime. Currently it
has the following constraints:
1. only works on KubeflowDagRunner;
2. only works for str-typed execution properties.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from typing import Any, Dict, Optional, Text, Type, Union

from tfx.utils import json_utils

# Regex pattern of RuntimeStringParameter.
PARAMETER_PATTERN = r'({{<RuntimeParameter>:.*?</RuntimeParameter>}})'

# Tag of RuntimeStringParameter placeholders.
PARAMETER_TAG = '{{<RuntimeParameter>: %s</RuntimeParameter>}}'

# Capturing regex to untag the RuntimeStringParameter placeholders.
PARAMETER_UNTAG = r'{{<RuntimeParameter>:(.*?)</RuntimeParameter>}}'


# TODO(b/144376447): Deprecate this once RuntimeParameter gets functional.
class RuntimeStringParameter(json_utils.Jsonable):
  """Runtime parameter represented by str-typed placeholder.

  Attributes:
    name: The name of the runtime parameter, serves as its unique identifier in
      the pipeline scope.
    default: Default value for runtime params when it's not explicitly
      specified.
    ptype: The type of the runtime parameter, restricted to Text for now.
    description: Description of the usage of the parameter
  """

  def __new__(
      cls,
      name: Optional[Text] = None,
      default: Optional[Union[int, float, bool, Text]] = None,
      ptype: Optional[Type] = None,  # pylint: disable=g-bare-generic
      description: Optional[Text] = None) -> Text:
    """Create a str-typed placeholder representing a pipeline parameter."""

    if ptype and ptype != Text:
      raise TypeError(
          'Currently only support string-typed placeholder parameters')
    if (default and ptype) and not isinstance(default, ptype):
      raise TypeError('Default value must be consistent with specified ptype')

    instance = object.__new__(cls)
    instance.name = name or ''
    instance.default = default
    instance.ptype = ptype
    instance.description = description

    return PARAMETER_TAG % json_utils.dumps(instance)

  def __init__(
      self,
      name: Text,
      default: Optional[Union[int, float, bool, Text]] = None,
      ptype: Optional[Type] = None,  # pylint: disable=g-bare-generic
      description: Optional[Text] = None):
    if ptype and ptype not in [int, float, bool, Text]:
      raise RuntimeError('Only str and scalar runtime parameters are supported')
    if (default and ptype) and not isinstance(default, ptype):
      raise TypeError('Default value must be consistent with specified ptype')
    self.name = name
    self.default = default
    self.ptype = ptype
    self.description = description

  # Need to override from_json_dict because __new__ actually returns a string
  # object.
  @classmethod
  def from_json_dict(
      cls,
      dict_data: Dict[Text, Any]) -> 'RuntimeStringParameter':
    """Convert from dictionary data to a RuntimeStringParameter."""
    instance = object.__new__(cls)
    instance.__dict__ = dict_data
    return instance

  @classmethod
  def parse(cls, placeholder: Text) -> 'RuntimeStringParameter':
    """Converts a placeholder to a RuntimeStringParameter obj.

    Args:
      placeholder: str-typed placeholder. Should be {{RuntimeParameter: ...}}

    Returns:
      A RuntimeStringParameter parsed from the placeholder
    """
    placeholder = placeholder.replace('\\', '')
    # Remove prefix and suffix.
    content = re.match(PARAMETER_UNTAG, placeholder)
    if content:
      return json_utils.loads(content.groups()[0])
    else:
      raise ValueError('Invalid RuntimeStringParameter placeholder found %s' %
                       placeholder)

  def __repr__(self):
    return ('RuntimeStringParam:\n  name: %s,\n  default: %s,\n  ptype: %s,\n  '
            'description: %s') % (self.name, self.default, self.ptype,
                                  self.description)

  def __eq__(self, other):
    return (isinstance(other.__class__, self.__class__) and
            self.name == other.name and self.default == other.default and
            self.ptype == other.ptype and self.description == other.description)
