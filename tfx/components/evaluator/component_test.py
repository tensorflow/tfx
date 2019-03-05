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
"""Tests for tfx.components.evaluator.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.evaluator import component
from tfx.proto import evaluator_pb2
from tfx.utils import channel
from tfx.utils import types


class ComponentTest(tf.test.TestCase):

  def test_construct(self):
    examples = types.TfxType(type_name='ExamplesPath')
    model_exports = types.TfxType(type_name='ModelExportPath')
    evaluator = component.Evaluator(
        examples=channel.as_channel([examples]),
        model_exports=channel.as_channel([model_exports]))
    self.assertEqual('ModelEvalPath', evaluator.outputs.output.type_name)

  def test_construct_with_slice_spec(self):
    examples = types.TfxType(type_name='ExamplesPath')
    model_exports = types.TfxType(type_name='ModelExportPath')
    evaluator = component.Evaluator(
        examples=channel.as_channel([examples]),
        model_exports=channel.as_channel([model_exports]),
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]))
    self.assertEqual('ModelEvalPath', evaluator.outputs.output.type_name)


if __name__ == '__main__':
  tf.test.main()
