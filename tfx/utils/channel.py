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
"""Definition of TFX Channel type.

Deprecated: please see the new location of this module at `tfx.types.channel`
and `tfx.types.channel_utils`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Iterable, List, Union, Text

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx import types
from tfx.types import channel_utils


@deprecation.deprecated(
    None, 'tfx.utils.types.Channel has been renamed to tfx.types.Channel as of '
    'TFX 0.14.0.')
class Channel(types.Channel):
  pass


@deprecation.deprecated(None,
                        'tfx.utils.channel.as_channel has been renamed to '
                        'tfx.types.channel_utils.as_channel as of TFX 0.14.0.')
def as_channel(source: Union[Channel, Iterable[types.Artifact]]) -> Channel:
  return channel_utils.as_channel(source)


@deprecation.deprecated(
    None, 'tfx.utils.channel.unwrap_channel_dict has been renamed to '
    'tfx.types.channel_utils.unwrap_channel_dict as of TFX 0.14.0.')
def unwrap_channel_dict(
    channel_dict: Dict[Text, Channel]) -> Dict[Text, List[types.Artifact]]:
  return channel_utils.unwrap_channel_dict(channel_dict)
