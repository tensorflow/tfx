# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Shared utility functions for ResolverOps."""

import functools
from typing import Dict, List, Optional, Sequence, Set

from tfx import types
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import filter_query_builder as q
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2


# Maps from "span" and "version" to PropertyType.INT. Many ResolverOps require
# one or both of these properties, so we define constants here for convenience.
SPAN_PROPERTY = {'span': types.artifact.PropertyType.INT}

VERSION_PROPERTY = {'version': types.artifact.PropertyType.INT}

SPAN_AND_VERSION_PROPERTIES = {**SPAN_PROPERTY, **VERSION_PROPERTY}

# Valid keys for the ResolverOp input/output dictionaries.
EXAMPLES_KEY = 'examples'
MODEL_KEY = 'model'
MODEL_BLESSSING_KEY = 'model_blessing'
MODEL_EXPORT_KEY = 'model_export'
MODEL_INFRA_BLESSING_KEY = 'model_infra_blessing'
MODEL_PUSH_KEY = 'model_push'
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'
ROOT_ARTIFACT_KEY = 'root_artifact'

# Taken from tfx.tflex.dsl.types.standard_artifacts. We don't use the existing
# constants due to Copybara.
EXAMPLES_TYPE_NAME = 'Examples'
TRANSFORM_GRAPH_TYPE_NAME = 'TransformGraph'
MODEL_TYPE_NAME = 'Model'
MODEL_RUN_TYPE_NAME = 'ModelRunPath'
MODEL_BLESSING_TYPE_NAME = 'ModelBlessing'
MODEL_INFRA_BLESSSING_TYPE_NAME = 'ModelInfraBlessingPath'
MODEL_PUSH_TYPE_NAME = 'ModelPushPath'

# Valid artifact TYPE_NAMEs by key.
ARTIFACT_TYPE_NAME_BY_KEY = {
    EXAMPLES_KEY: EXAMPLES_TYPE_NAME,
    MODEL_KEY: MODEL_TYPE_NAME,
    MODEL_BLESSSING_KEY: MODEL_BLESSING_TYPE_NAME,
    MODEL_INFRA_BLESSING_KEY: MODEL_INFRA_BLESSSING_TYPE_NAME,
    MODEL_PUSH_KEY: MODEL_PUSH_TYPE_NAME,
}

# Maximum number of hops for lineage tracing using MLMD metadata_resolver.

# Set to 100 because this is the maximum value allowed by MLMD.
GRAPH_TRAVERSAL_OP_MAX_NUM_HOPS = 100

# Set to 50 because we assume a Model is relatively few hops away from
# ModelBlessing, ModelInfraBlessing, and ModelPush.
LATEST_POLICY_MODEL_OP_MAX_NUM_HOPS = 50

# A fixed batch size for batch querying APIs in MLMD metadata_resolver.
BATCH_SIZE = 100


def get_valid_artifact_states_filter_query(
    valid_artifact_states: Sequence['metadata_store_pb2.Artifact.State'],
) -> str:
  """Returns a filter query for valid artifact states."""
  return f'state IN {q.to_sql_string(valid_artifact_states)}'


# TODO(b/269147946): Put validation logic inside ResolverOp Property instead,
# providing compile time check.
def validate_argument(
    argument_name: str,
    argument_value: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
):
  """Validates that the argument is >= a min value and/or <= a max value."""
  if min_value is not None and argument_value < min_value:
    raise exceptions.InvalidArgument(
        f'{argument_name} must be >= {min_value} but was set to'
        f' {argument_value}.'
    )

  if max_value is not None and argument_value > max_value:
    raise exceptions.InvalidArgument(
        f'{argument_name} must be <= {max_value} but was set to'
        f' {argument_value}.'
    )


def validate_input_dict(
    input_dict: typing_utils.ArtifactMultiMap,
    valid_keys: Set[str],
    requires_all: bool = False,
):
  """Checks that the input_dict is properly formatted."""
  if requires_all and set(input_dict.keys()) != valid_keys:
    raise exceptions.InvalidArgument(
        f'input_dict must have all keys in {valid_keys}, but only had '
        f'{input_dict.keys()}'
    )

  for key in input_dict.keys():
    if key not in valid_keys:
      raise exceptions.InvalidArgument(
          'input_dict can only have keys {valid_keys}, but contained'
          f' key {key}.'
      )

    for artifact in input_dict[key]:
      if artifact.TYPE_NAME != ARTIFACT_TYPE_NAME_BY_KEY[key]:
        raise exceptions.InvalidArgument(
            f'Artifacts of input_dict["{key}"] are expected to have artifacts'
            f' with TYPE_NAME {ARTIFACT_TYPE_NAME_BY_KEY[key]}, but'
            f' artifact {artifact} in input_dict["{key}"] had TYPE_NAME'
            f' {artifact.TYPE_NAME}.'
        )


def get_valid_artifacts(
    artifacts: Sequence[types.Artifact],
    property_types: Dict[str, types.artifact.PropertyType],
) -> List[types.Artifact]:
  """Returns artifacts that have the required property names and types."""

  valid_artifacts = []
  for artifact in artifacts:
    if artifact.PROPERTIES is None:
      continue

    for property_name, property_type in property_types.items():
      if (
          property_name not in artifact.PROPERTIES
          or artifact.PROPERTIES[property_name].type != property_type
      ):
        break
    else:
      valid_artifacts.append(artifact)

  return valid_artifacts


def filter_artifacts_by_span(
    artifacts: List[types.Artifact],
    span_descending: bool,
    n: int = 1,
    skip_last_n: int = 0,
    keep_all_versions: bool = False,
    min_span: Optional[int] = None,
    version_sort_keys: Sequence[str] = (),
) -> List[types.Artifact]:
  """Filters artifacts by their "span" PROPERTY.

  This should only be used a shared utility for LatestSpan and ConsecutiveSpans.

  Args:
    artifacts: The list of Artifacts to filter.
    span_descending: If true, then the artifacts will be sorted by span in
      descending order.  Else, they will be sorted in ascending order by span.
      Set to true for LatestSpan, and set to false for ConsecutiveSpans.
    n: The number of spans to return. If n <= 0, then n is set to the total
      number of unique spans.
    skip_last_n: Number of largest spans to skip. For example, if the spans are
      [1, 2, 3] and skip_last_n=1, then only spans [1, 2] will be considered.
    keep_all_versions: If true, all versions of the n spans are returned. Else,
      only the latest version is returned.
    min_span: Minimum span before which no span will be considered.
    version_sort_keys: List of string artifact attributes to sort or filter the
      versions witin the spans, applied in order of specification. Nested keys
      can use '.' separator for e.g. 'mlmd_artifact.create_time_since_epoch'.
      The default key is version number, create time and id in the same order.

  Returns:
    The filtered artifacts.
  """
  if not artifacts:
    return []

  # Only keep artifacts with spans >= min_span and account for skip_last_n
  spans = sorted({a.span for a in artifacts})
  if min_span is not None:
    spans = [s for s in spans if s >= min_span]
  if skip_last_n:
    spans = spans[:-skip_last_n]

  # Sort spans in descending order, if specified.
  if span_descending:
    spans = spans[::-1]

  # Keep n spans, if n is positive.
  if n > 0:
    spans = spans[:n]

  if not spans:
    return []

  artifacts_by_span = {}
  for artifact in artifacts:
    artifacts_by_span.setdefault(artifact.span, []).append(artifact)

  if version_sort_keys:
    # Recursively resolve nested key attributes like
    # 'mlmd_artifact.create_time_since_epoch' to the form
    # getattr(getattr(artifact, 'mlmd_artifact'), 'create_time_since_epoch')
    key = lambda a: (
        tuple(
            functools.reduce(getattr, k.split('.'), a)
            for k in version_sort_keys
        )
    )
  else:
    # span_descending only applies to sorting by span, but version should
    # always be sorted in ascending order. By default, latest version is defined
    # as the largest version and ties are broken by create_time and  id.
    key = lambda a: (  # pylint: disable=g-long-lambda
        a.version,
        a.mlmd_artifact.create_time_since_epoch,
        a.id,
    )

  result = []
  for span in sorted(spans):
    if keep_all_versions:
      result.extend(sorted(artifacts_by_span[span], key=key))
    else:
      result.append(max(artifacts_by_span[span], key=key))

  return result


def sort_artifact_dict(
    artifacts_by_key: Dict[str, List[types.Artifact]]
) -> Dict[str, List[types.Artifact]]:
  """Sorts the artifact dict values by (creation time, id)."""
  for key, artifacts in artifacts_by_key.items():
    artifacts_by_key[key] = sorted(
        artifacts,
        # If the user wants to sort Examples artifacts by span/version, they
        # can call the all_spans(...) canned resolver functions.
        key=lambda a: (a.mlmd_artifact.create_time_since_epoch, a.id),
    )
  return artifacts_by_key
