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
"""Tests for tfx.components.trainer.fn_args_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tfx.components.trainer import fn_args_utils
from tfx.proto import trainer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import proto_utils


class FnArgsUtilsTest(tf.test.TestCase):

  def testGetCommonFnArgs(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir,
                                'transform/transformed_examples')
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    transform_output = standard_artifacts.TransformGraph()
    transform_output.uri = os.path.join(source_data_dir,
                                        'transform/transform_graph')

    schema = standard_artifacts.Schema()
    schema.uri = os.path.join(source_data_dir, 'schema_gen')

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.TRANSFORM_GRAPH_KEY: [transform_output],
        standard_component_specs.SCHEMA_KEY: [schema],
    }

    # Create exec properties skeleton.
    exec_properties = {
        'train_args':
            proto_utils.proto_to_json(trainer_pb2.TrainArgs(num_steps=1000)),
        'eval_args':
            proto_utils.proto_to_json(trainer_pb2.EvalArgs(num_steps=500)),
    }

    fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
                                               'tempdir')
    self.assertEqual(fn_args.working_dir, 'tempdir')
    self.assertEqual(fn_args.train_steps, 1000)
    self.assertEqual(fn_args.eval_steps, 500)
    self.assertLen(fn_args.train_files, 1)
    self.assertEqual(fn_args.train_files[0],
                     os.path.join(examples.uri, 'Split-train', '*'))
    self.assertLen(fn_args.eval_files, 1)
    self.assertEqual(fn_args.eval_files[0],
                     os.path.join(examples.uri, 'Split-eval', '*'))
    self.assertEqual(fn_args.schema_path,
                     os.path.join(schema.uri, 'schema.pbtxt'))
    self.assertEqual(fn_args.transform_graph_path, transform_output.uri)
    self.assertIsInstance(fn_args.data_accessor, fn_args_utils.DataAccessor)


if __name__ == '__main__':
  tf.test.main()
