# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Predicate for Conditional channels."""

from typing import Any, Dict, Callable
from tfx.utils import json_utils

# Not importing tfx.types, because it will cause circular dependencies later on.
# tfx.types -> predicate -> placeholder -> channels -> tfx.types
tfx_types = Any


class PredicateWithInputKeys:

  def to_json_dict(self) -> Dict[str, Any]:
    return {}


class Predicate(json_utils.Jsonable):
  """Experimental Predicate object."""

  def replace_placeholder_keys(
      self, channel_to_key_fn: Callable[['tfx_types.Channel'], str]
  ) -> PredicateWithInputKeys:
    """Replaces the placeholder key, enabling Predicate resolution.

    Predicate objects are created during Pipeline authoring time, composed from
    output Channels of upstream nodes, whose future values are represented by
    ArtifactPlaceholders. These values are available at input resolution time,
    and at that time, they can be retrieved from a dictionary by supplying the
    right input channel key.

    However:
    1) Different nodes that map to the same Predicate might use different input
       keys to refer to the same underlying Channel.
    2) When a Predicate object is first constructed, it doesn't know which nodes
       are going to use it.

    Thus, the ArtifactPlaceholders that represent the Channels' future values
    are initially constructed without any "key". The "keys" will be "filled in"
    only at compilation time -- when the compiler is converting the DSL Node
    object into IR, the Predicate objects for that node will have their
    underlying ArtifactPlaceholders be filled in with their respective input
    keys. Consequently, each node's IR will contain the right input key for
    obtain the Channel's future value, enabling it to resolve the Predicate.

    To do so, the compiler implements a function `channel_to_key_fn`, which maps
    a Channel to the input key for retrieving the future values at input
    resolution time.

    When `replace_placeholder_keys` method is called on a Predicate
    instance with the `channel_to_key_fn`, it will traverse its internal data
    structure, and whenever it finds a ArtifactPlaceholder, it will retrieve the
    associated Channel, call `channel_to_key_fn` with it, and use the returned
    key to replace the ArtifactPlaceholder's key.

    Args:
      channel_to_key_fn: A function that takes a Channel and produces the input
        key. Note that the channel_to_key_fn for different nodes would likely be
        different, since different nodes can use different input keys to refer
        to the same Channel during input resolution.

    Returns:
      A copy of the Predicate, with the ArtifactPlaceholders' keys filled in.
    """
    del channel_to_key_fn
    return PredicateWithInputKeys()
