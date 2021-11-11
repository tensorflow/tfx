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
"""Utilities for executor of Transform."""

import functools
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from absl import logging

from tfx import types
from tfx.components.transform import labels
from tfx.components.util import value_utils
from tfx.proto import transform_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import json_utils
from tfx.utils import proto_utils


# Default file name prefix for transformed_examples.
_DEFAULT_TRANSFORMED_EXAMPLES_PREFIX = 'transformed_examples'


def MaybeBindCustomConfig(inputs: Mapping[str, Any],
                          fn: Any) -> Callable[..., Any]:
  # For compatibility, only bind custom config if it's in the signature.
  if value_utils.FunctionHasArg(fn, labels.CUSTOM_CONFIG):
    custom_config_json = value_utils.GetSoleValue(inputs, labels.CUSTOM_CONFIG)
    custom_config = (json_utils.loads(custom_config_json)
                     if custom_config_json else {}) or {}
    fn = functools.partial(fn, custom_config=custom_config)
  return fn


def ValidateOnlyOneSpecified(inputs: Mapping[str, Any],
                             keys: Sequence[str],
                             allow_missing: bool = False) -> bool:
  """Check whether only one of given keys are specified in the input.

  NOTE: False-equivalent values like 0, '' are regarded as not specified.

  Args:
    inputs: input dictionary.
    keys: keys to check the existence of values.
    allow_missing: If False, one of keys should be set in inputs.

  Returns:
    True if one of the key has a value.
  Raises:
    ValueError: if none of the keys have non empty value in the input.
  """
  counter = 0
  for key in keys:
    counter += int(bool(value_utils.GetSoleValue(inputs, key, strict=False)))

  keys_str = ', '.join(keys)
  if counter > 1:
    raise ValueError(
        f'At most one of {keys_str} should be supplied in the input.')
  elif counter == 0 and not allow_missing:
    raise ValueError(f'One of {keys_str} should be supplied in the input.')

  return counter > 0


def MatchNumberOfTransformedExamplesArtifacts(
    input_dict: Dict[str, List[types.Artifact]],
    output_dict: Dict[str, List[types.Artifact]]) -> None:
  """Alters output_dict to have the same number of examples to input.

  If there are multiple input Examples artifacts, replicate Examples artifact
  in output_dict to have the same number of artifacts. The resulting artifact
  will have URIs that is located under the original output uri.
  No-op if there is one or less Examples artifact in the input_dict.

  Args:
    input_dict: input artifact dictionary of the Executor.
    output_dict: output artifact dictionary of the Executor.
  """
  num_examples = len(input_dict[standard_component_specs.EXAMPLES_KEY])
  if (num_examples > 1 and
      standard_component_specs.TRANSFORMED_EXAMPLES_KEY in output_dict and
      len(output_dict[standard_component_specs.TRANSFORMED_EXAMPLES_KEY]) == 1):
    output_dict[standard_component_specs
                .TRANSFORMED_EXAMPLES_KEY] = artifact_utils.replicate_artifacts(
                    output_dict[
                        standard_component_specs.TRANSFORMED_EXAMPLES_KEY][0],
                    num_examples)


def ResolveSplitsConfig(
    splits_config_str: Optional[str],
    examples: List[types.Artifact]) -> transform_pb2.SplitsConfig:
  """Resolve SplitsConfig proto for the transfrom request."""
  result = transform_pb2.SplitsConfig()
  if splits_config_str:
    proto_utils.json_to_proto(splits_config_str, result)
    if not result.analyze:
      raise ValueError('analyze cannot be empty when splits_config is set.')
    return result

  result.analyze.append('train')

  # All input artifacts should have the same set of split names.
  split_names = set(artifact_utils.decode_split_names(examples[0].split_names))

  for artifact in examples:
    artifact_split_names = set(
        artifact_utils.decode_split_names(artifact.split_names))
    if split_names != artifact_split_names:
      raise ValueError(
          'Not all input artifacts have the same split names: (%s, %s)' %
          (split_names, artifact_split_names))

  result.transform.extend(split_names)
  logging.info("Analyze the 'train' split and transform all splits when "
               'splits_config is not set.')
  return result


def SetSplitNames(
    splits: Sequence[str],
    transformed_examples: Optional[List[types.Artifact]]) -> None:
  """Sets split_names property of input artifacts."""
  if not transformed_examples:
    return

  for artifact in transformed_examples:
    artifact.split_names = artifact_utils.encode_split_names(list(splits))


def GetSplitPaths(
    transformed_examples: Optional[List[types.Artifact]]) -> List[str]:
  """Gets all paths for splits in the input artifacts."""
  result = []
  if not transformed_examples:
    return result
  splits = artifact_utils.decode_split_names(
      transformed_examples[0].split_names)

  for split in splits:
    transformed_example_uris = artifact_utils.get_split_uris(
        transformed_examples, split)
    for output_uri in transformed_example_uris:
      result.append(
          os.path.join(output_uri, _DEFAULT_TRANSFORMED_EXAMPLES_PREFIX))

  return result


def GetCachePathEntry(
    label: str, params_dict: Dict[str, List[types.Artifact]]) -> Dict[str, str]:
  """Returns a cachePath entry if label exists in params_dict."""
  # Covers the cases: path wasn't provided, or was provided an empty list.
  if not params_dict.get(label):
    return {}

  if label == standard_component_specs.ANALYZER_CACHE_KEY:
    dict_key = labels.CACHE_INPUT_PATH_LABEL
  elif label == standard_component_specs.UPDATED_ANALYZER_CACHE_KEY:
    dict_key = labels.CACHE_OUTPUT_PATH_LABEL
  return {dict_key: artifact_utils.get_single_uri(params_dict[label])}


def GetStatsOutputPathEntries(
    disable_statistics: bool,
    output_dict: Dict[str, List[types.Artifact]]) -> Dict[str, str]:
  """Returns output entries for stats output path."""
  label_component_key_list = [
      (labels.PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL,
       standard_component_specs.PRE_TRANSFORM_STATS_KEY),
      (labels.PRE_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL,
       standard_component_specs.PRE_TRANSFORM_SCHEMA_KEY),
      (labels.POST_TRANSFORM_OUTPUT_ANOMALIES_PATH_LABEL,
       standard_component_specs.POST_TRANSFORM_ANOMALIES_KEY),
      (labels.POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL,
       standard_component_specs.POST_TRANSFORM_STATS_KEY),
      (labels.POST_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL,
       standard_component_specs.POST_TRANSFORM_SCHEMA_KEY)
  ]
  result = {}
  if not disable_statistics:
    for label, component_key in label_component_key_list:
      if component_key in output_dict:
        result[label] = artifact_utils.get_single_uri(
            output_dict[component_key])
  if result and len(result) != len(label_component_key_list):
    raise ValueError(
        'Either all stats_output_paths should be specified or none.')
  return result
