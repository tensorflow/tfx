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

from typing import Any, Callable, Dict, List, Optional, Text, NamedTuple

import absl
import tensorflow as tf
from tfx import types
from tfx.components.trainer import constants
from tfx.components.util import tfxio_utils
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from google.protobuf import json_format
from tensorflow_metadata.proto.v0 import schema_pb2


_TELEMETRY_DESCRIPTORS = ['Trainer']

# TODO(b/162532757): the type should be
# tfx_bsl.tfxio.dataset_options.TensorFlowDatasetOptions. Switch to it once
# tfx-bsl post-0.22 is released.
_TensorFlowDatasetOptions = Any

DataAccessor = NamedTuple('DataAccessor', [
    ('tf_dataset_factory', Callable[[
        List[Text],
        _TensorFlowDatasetOptions,
        Optional[schema_pb2.Schema],
    ], tf.data.Dataset])
])

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
# data_accessor: Contains factories that can create tf.data.Datasets or other
#   means to access the train/eval data. They provide a uniform way of
#   accessing data, regardless of how the data is stored on disk.
# custom_config: An optional dictionary passed to the component.
FnArgs = NamedTuple('FnArgs', [
    ('working_dir', Text),
    ('train_files', List[Text]),
    ('eval_files', List[Text]),
    ('train_steps', int),
    ('eval_steps', int),
    ('schema_path', Text),
    ('transform_graph_path', Text),
    ('data_accessor', DataAccessor),
    ('custom_config', Dict[Text, Any]),
])
# Set default value to None.
FnArgs.__new__.__defaults__ = (None,) * len(FnArgs._fields)


def get_common_fn_args(input_dict: Dict[Text, List[types.Artifact]],
                       exec_properties: Dict[Text, Any],
                       working_dir: Text = None) -> FnArgs:
  """Get common args of training and tuning."""
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

  # Default behavior is train on `train` split (when splits is empty in train
  # args) and evaluate on `eval` split (when splits is empty in eval args).
  if not train_args.splits:
    train_args.splits.append('train')
    absl.logging.info("Train on the 'train' split when train_args.splits is "
                      'not set.')
  if not eval_args.splits:
    eval_args.splits.append('eval')
    absl.logging.info("Evaluate on the 'eval' split when eval_args.splits is "
                      'not set.')

  train_files = []
  for train_split in train_args.splits:
    train_files.extend([
        io_utils.all_files_pattern(uri)
        for uri in artifact_utils.get_split_uris(
            input_dict[constants.EXAMPLES_KEY], train_split)
    ])

  eval_files = []
  for eval_split in eval_args.splits:
    eval_files.extend([
        io_utils.all_files_pattern(uri)
        for uri in artifact_utils.get_split_uris(
            input_dict[constants.EXAMPLES_KEY], eval_split)
    ])

  data_accessor = DataAccessor(
      tf_dataset_factory=tfxio_utils.get_tf_dataset_factory_from_artifact(
          input_dict[constants.EXAMPLES_KEY], _TELEMETRY_DESCRIPTORS))

  # https://github.com/tensorflow/tfx/issues/45: Replace num_steps=0 with
  # num_steps=None.  Conversion of the proto to python will set the default
  # value of an int as 0 so modify the value here.  Tensorflow will raise an
  # error if num_steps <= 0.
  train_steps = train_args.num_steps or None
  eval_steps = eval_args.num_steps or None

  # TODO(b/156929910): Refactor Trainer to be consistent with empty or None
  #                    custom_config handling.
  custom_config = json_utils.loads(
      exec_properties.get(constants.CUSTOM_CONFIG_KEY, 'null'))

  return FnArgs(
      working_dir=working_dir,
      train_files=train_files,
      eval_files=eval_files,
      train_steps=train_steps,
      eval_steps=eval_steps,
      schema_path=schema_path,
      transform_graph_path=transform_graph_path,
      data_accessor=data_accessor,
      custom_config=custom_config,
  )
