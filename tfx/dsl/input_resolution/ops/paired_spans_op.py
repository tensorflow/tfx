# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Module for PairedSpans operator."""


import functools
from typing import Dict, Sequence, Tuple

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.utils import typing_utils


class PairedSpans(
    resolver_op.ResolverOp,
    canonical_name='tfx.PairedSpans',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """PairedSpans operator."""

  # If true, spans are paired across all versions. If False, only the latest
  # versioned spans are paired up.
  keep_all_versions = resolver_op.Property(type=bool, default=False)

  def _get_artifacts(
      self, artifacts: Sequence[types.Artifact]
  ) -> Dict[Tuple[int, int], types.Artifact]:
    valid = ops_utils.get_valid_artifacts(
        artifacts, ops_utils.SPAN_AND_VERSION_PROPERTIES
    )
    artifact_dict = {}
    for art in ops_utils.filter_artifacts_by_span(
        valid,
        span_descending=False,
        n=0,
        keep_all_versions=self.keep_all_versions,
    ):
      artifact_dict[(art.span, art.version)] = art
    return artifact_dict

  def apply(self, input_dict: typing_utils.ArtifactMultiMap):
    """PairedSpans Operator.

    This is similiar to the unnest operator, but will provide a list of paired
    artifacts by span and version over each key of the input dict. Pseudo code
    example where matching subscripts indicate matching span number and version
    number.

    Notation here is {artifact}_(span, version)

      PairedSpans({x: [x_(0,0), x_(0,1), x_(1,0), x_(2,0)],
                   y: [y_(0,0), y_(0,1), y_(1,0)]})
        = [{x: [x_(0,0)], y: [y_(0,0)]},
           {x: [x_(0,1)], y: [y_(0,1)]},
           {x: [x_(1,0)], y: [y_(1,0)]}]

      PairedSpans({x: [x_(0,0), x_(0,1), x_(1,0), x_(2,0)],
                   y: [y_(0,0), y_(0,1), y_(1,0)]},
                   keep_all_versions=False)
        = [{x: [x_(0,1)], y: [y_(0,1)]},
           {x: [x_(1,0)], y: [y_(1,0)]}]



    Args:
      input_dict: A dictionary of artifacts.
    Returns:
      List of dicts of paired input elements with the same span and version.
    """
    indexed_latest_artifacts = {
        k: self._get_artifacts(v) for k, v in input_dict.items()
    }
    if not indexed_latest_artifacts:
      return []

    if len(indexed_latest_artifacts) < 2:
      return []
    # Intersect all sets of keys.
    common_keys = functools.reduce(
        lambda x, y: x & y,
        [set(x.keys()) for x in indexed_latest_artifacts.values()],
    )

    return [
        {ak: [av[ck]] for ak, av in indexed_latest_artifacts.items()}
        for ck in sorted(common_keys)
    ]
