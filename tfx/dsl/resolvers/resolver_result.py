# Lint as: python3
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
"""Result for TFX resolvers."""
from typing import Dict, List, Text

from tfx import types


class ResolveResult(object):
  """The data structure to hold results from Resolver.

  Attributes:
    per_key_resolve_result: a key -> List[Artifact] dict containing the resolved
      artifacts for each source channel with the key as tag.
    per_key_resolve_state: a key -> bool dict containing whether or not the
      resolved artifacts for the channel are considered complete.
    has_complete_result: bool value indicating whether all desired artifacts
      have been resolved.
  """

  def __init__(self, per_key_resolve_result: Dict[Text, List[types.Artifact]],
               per_key_resolve_state: Dict[Text, bool]):
    self.per_key_resolve_result = per_key_resolve_result
    self.per_key_resolve_state = per_key_resolve_state
    self.has_complete_result = all(list(per_key_resolve_state.values()))

