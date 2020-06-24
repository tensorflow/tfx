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
"""FnArgs for passing information to UDF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text, NamedTuple

from google.protobuf import json_format

from tfx import types
from tfx.components.trainer import constants
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils

# TODO(b/156929910): Change TrainerFnArgs to this FnArgs.
#
# working_dir: Working dir.
# train_files: A list of patterns for train files.
# eval_files: A list of patterns for eval files.
# train_steps: Number of train steps.
# eval_steps: Number of eval steps.
# schema_path: A single uri for schema file. Will be None if not specified.
# transform_graph_path: An optional single uri for transform graph produced by
#                       TFT. Will be None if not specified.
FnArgs = NamedTuple('FnArgs', [('working_dir', Text),
                               ('train_files', List[Text]),
                               ('eval_files', List[Text]), ('train_steps', int),
                               ('eval_steps', int), ('schema_path', Text),
                               ('transform_graph_path', Text)])
# Set default value to None.
FnArgs.__new__.__defaults__ = (None,) * len(FnArgs._fields)


def get_common_fn_args(input_dict: Dict[Text, List[types.Artifact]],
                       exec_properties: Dict[Text, Any],
                       working_dir: Text = None) -> FnArgs:
  """Get common args of training and tuning."""
  train_files = [
      io_utils.all_files_pattern(
          artifact_utils.get_split_uri(input_dict[constants.EXAMPLES_KEY],
                                       'train'))
  ]
  eval_files = [
      io_utils.all_files_pattern(
          artifact_utils.get_split_uri(input_dict[constants.EXAMPLES_KEY],
                                       'eval'))
  ]

  if input_dict.get(constants.TRANSFORM_GRAPH_KEY):
    transform_graph_path = artifact_utils.get_single_uri(
        input_dict[constants.TRANSFORM_GRAPH_KEY])
  else:
    transform_graph_path = None

  if input_dict.get(constants.SCHEMA_KEY):
    schema_path = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict[constants.SCHEMA_KEY]))
  else:
    schema_path = None

  train_args = trainer_pb2.TrainArgs()
  eval_args = trainer_pb2.EvalArgs()
  json_format.Parse(exec_properties[constants.TRAIN_ARGS_KEY], train_args)
  json_format.Parse(exec_properties[constants.EVAL_ARGS_KEY], eval_args)

  # https://github.com/tensorflow/tfx/issues/45: Replace num_steps=0 with
  # num_steps=None.  Conversion of the proto to python will set the default
  # value of an int as 0 so modify the value here.  Tensorflow will raise an
  # error if num_steps <= 0.
  train_steps = train_args.num_steps or None
  eval_steps = eval_args.num_steps or None

  return FnArgs(
      working_dir=working_dir,
      train_files=train_files,
      eval_files=eval_files,
      train_steps=train_steps,
      eval_steps=eval_steps,
      schema_path=schema_path,
      transform_graph_path=transform_graph_path,
  )
