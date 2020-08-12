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
"""Tests for tfx.components.evaluator.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text
import tensorflow as tf
import tensorflow_model_analysis as tfma

from tfx.components.evaluator import component
from tfx.orchestration import data_types
from tfx.proto import evaluator_pb2
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        example_splits=['eval'])
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)
    self.assertEqual(standard_artifacts.ModelBlessing.TYPE_NAME,
                     evaluator.outputs['blessing'].type_name)
    self.assertEqual(
        json_utils.dumps(['eval']), evaluator.exec_properties['example_splits'])

  def testConstructWithBaselineModel(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    baseline_model = standard_artifacts.Model()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        baseline_model=channel_utils.as_channel([baseline_model]))
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)

  def testConstructWithSliceSpec(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]))
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)

  def testConstructWithFairnessThresholds(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]),
        fairness_indicator_thresholds=[0.1, 0.3, 0.5, 0.9])
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)

  def testConstructWithParameter(self):
    column_name = data_types.RuntimeParameter(name='column-name', ptype=Text)
    threshold = data_types.RuntimeParameter(name='threshold', ptype=float)
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        feature_slicing_spec={'specs': [{
            'column_for_slicing': [column_name]
        }]},
        fairness_indicator_thresholds=[threshold])
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)

  def testConstructWithEvalConfig(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    schema = standard_artifacts.Schema()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model_exports=channel_utils.as_channel([model_exports]),
        eval_config=tfma.EvalConfig(
            slicing_specs=[tfma.SlicingSpec(feature_keys=['trip_start_hour'])]),
        schema=channel_utils.as_channel([schema]),)
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)

  def testConstructWithModuleFile(self):
    examples = standard_artifacts.Examples()
    model_exports = standard_artifacts.Model()
    evaluator = component.Evaluator(
        examples=channel_utils.as_channel([examples]),
        model=channel_utils.as_channel([model_exports]),
        example_splits=['eval'],
        module_file='path')
    self.assertEqual(standard_artifacts.ModelEvaluation.TYPE_NAME,
                     evaluator.outputs['evaluation'].type_name)
    self.assertEqual('path', evaluator.exec_properties['module_file'])


if __name__ == '__main__':
  tf.test.main()
