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

from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Text

import absl
import attr
import pyarrow as pa
import tensorflow as tf
from tfx import types
from tfx.components.util import tfxio_utils
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import proto_utils
from tfx_bsl.tfxio import dataset_options

from tensorflow_metadata.proto.v0 import schema_pb2


_TELEMETRY_DESCRIPTORS = ['Trainer']

DataAccessor = NamedTuple('DataAccessor',
                          [('tf_dataset_factory', Callable[[
                              List[Text],
                              dataset_options.TensorFlowDatasetOptions,
                              Optional[schema_pb2.Schema],
                          ], tf.data.Dataset]),
                           ('record_batch_factory', Callable[[
                               List[Text],
                               dataset_options.RecordBatchesOptions,
                               Optional[schema_pb2.Schema],
                           ], Iterator[pa.RecordBatch]])])


@attr.s
class FnArgs:
  """Args to pass to user defined training/tuning function(s).

  Attributes:
    working_dir: Working dir.
    train_files: A list of patterns for train files.
    eval_files: A list of patterns for eval files.
    train_steps: Number of train steps.
    eval_steps: Number of eval steps.
    schema_path: A single uri for schema file. Will be None if not specified.
    schema_file: Deprecated, use `schema_path` instead.
    transform_graph_path: An optional single uri for transform graph produced by
      TFT. Will be None if not specified.
    transform_output: Deprecated, use `transform_graph_path` instead.'
    data_accessor: Contains factories that can create tf.data.Datasets or other
      means to access the train/eval data. They provide a uniform way of
      accessing data, regardless of how the data is stored on disk.
    serving_model_dir: A single uri for the output directory of the serving
      model.
    eval_model_dir: A single uri for the output directory of the eval model.
      Note that this is estimator only, Keras doesn't require it for TFMA.
    model_run_dir: A single uri for the output directory of model training
      related files.
    base_model: An optional base model path that will be used for this training.
    hyperparameters: An optional kerastuner.HyperParameters config.
    custom_config: An optional dictionary passed to the component.
  """
  working_dir = attr.ib(type=Text, default=None)
  train_files = attr.ib(type=List[Text], default=None)
  eval_files = attr.ib(type=List[Text], default=None)
  train_steps = attr.ib(type=int, default=None)
  eval_steps = attr.ib(type=int, default=None)
  schema_path = attr.ib(type=Text, default=None)
  schema_file = attr.ib(type=Text, default=None)
  transform_graph_path = attr.ib(type=Text, default=None)
  transform_output = attr.ib(type=Text, default=None)
  data_accessor = attr.ib(type=DataAccessor, default=None)
  serving_model_dir = attr.ib(type=Text, default=None)
  eval_model_dir = attr.ib(type=Text, default=None)
  model_run_dir = attr.ib(type=Text, default=None)
  base_model = attr.ib(type=Text, default=None)
  hyperparameters = attr.ib(type=Text, default=None)
  custom_config = attr.ib(type=Dict[Text, Any], default=None)


def get_common_fn_args(input_dict: Dict[Text, List[types.Artifact]],
                       exec_properties: Dict[Text, Any],
                       working_dir: Text = None) -> FnArgs:
  """Get common args of training and tuning."""
  if input_dict.get(standard_component_specs.TRANSFORM_GRAPH_KEY):
    transform_graph_path = artifact_utils.get_single_uri(
        input_dict[standard_component_specs.TRANSFORM_GRAPH_KEY])
  else:
    transform_graph_path = None

  if input_dict.get(standard_component_specs.SCHEMA_KEY):
    schema_path = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(
            input_dict[standard_component_specs.SCHEMA_KEY]))
  else:
    schema_path = None

  train_args = trainer_pb2.TrainArgs()
  eval_args = trainer_pb2.EvalArgs()
  proto_utils.json_to_proto(
      exec_properties[standard_component_specs.TRAIN_ARGS_KEY], train_args)
  proto_utils.json_to_proto(
      exec_properties[standard_component_specs.EVAL_ARGS_KEY], eval_args)

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
            input_dict[standard_component_specs.EXAMPLES_KEY], train_split)
    ])

  eval_files = []
  for eval_split in eval_args.splits:
    eval_files.extend([
        io_utils.all_files_pattern(uri)
        for uri in artifact_utils.get_split_uris(
            input_dict[standard_component_specs.EXAMPLES_KEY], eval_split)
    ])

  data_accessor = DataAccessor(
      tf_dataset_factory=tfxio_utils.get_tf_dataset_factory_from_artifact(
          input_dict[standard_component_specs.EXAMPLES_KEY],
          _TELEMETRY_DESCRIPTORS),
      record_batch_factory=tfxio_utils.get_record_batch_factory_from_artifact(
          input_dict[standard_component_specs.EXAMPLES_KEY],
          _TELEMETRY_DESCRIPTORS))

  # https://github.com/tensorflow/tfx/issues/45: Replace num_steps=0 with
  # num_steps=None.  Conversion of the proto to python will set the default
  # value of an int as 0 so modify the value here.  Tensorflow will raise an
  # error if num_steps <= 0.
  train_steps = train_args.num_steps or None
  eval_steps = eval_args.num_steps or None

  # Load and deserialize custom config from execution properties.
  # Note that in the component interface the default serialization of custom
  # config is 'null' instead of '{}'. Therefore we need to default the
  # json_utils.loads to 'null' then populate it with an empty dict when
  # needed.
  custom_config = json_utils.loads(
      exec_properties.get(standard_component_specs.CUSTOM_CONFIG_KEY, 'null'))

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
