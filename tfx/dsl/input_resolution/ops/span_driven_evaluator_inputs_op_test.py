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
"""Tests for tfx.dsl.input_resolution.ops.span_driven_evaluator_inputs_op."""
from typing import List, Optional

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions


# TODO(b/269144878): Refactor these tests to have a helper method check() to
# compare expected and actual dictionaries.
class SpanDrivenEvaluatorInputsOpTest(
    test_utils.ResolverTestCase,
):

  def _span_driven_evaluator(
      self,
      models: Optional[List[types.Artifact]] = None,
      examples: Optional[List[types.Artifact]] = None,
      **kwargs
  ):
    input_dict = {
        ops_utils.MODEL_KEY: models,
        ops_utils.EXAMPLES_KEY: examples,
    }
    return self._run_span_driven_evaluator(input_dict, **kwargs)

  def _run_span_driven_evaluator(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.SpanDrivenEvaluatorInputs,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    # We intentionally save a variable of each Examples/Model artifact so that
    # the tests are more readable.
    self.examples_1 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 1, 'version': 1}
    )
    self.examples_2 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 2, 'version': 1}
    )
    self.examples_3 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 3, 'version': 1}
    )
    self.examples_4 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 4, 'version': 1}
    )
    self.examples_5 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 5, 'version': 1}
    )
    self.examples_6 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 6, 'version': 1}
    )
    self.examples_7 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 7, 'version': 1}
    )
    self.examples_8 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 8, 'version': 1}
    )
    self.examples_9 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 9, 'version': 1}
    )
    self.examples_10 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 10, 'version': 1}
    )
    self.examples = [
        self.examples_1,
        self.examples_2,
        self.examples_3,
        self.examples_4,
        self.examples_5,
        self.examples_6,
        self.examples_7,
        self.examples_8,
        self.examples_9,
        self.examples_10,
    ]

    transform_graph = self.prepare_tfx_artifact(test_utils.TransformGraph)

    self.model_1 = self.prepare_tfx_artifact(test_utils.Model)
    self.model_2 = self.prepare_tfx_artifact(test_utils.Model)
    self.model_3 = self.prepare_tfx_artifact(test_utils.Model)
    self.model_4 = self.prepare_tfx_artifact(test_utils.Model)
    self.model_5 = self.prepare_tfx_artifact(test_utils.Model)
    self.models = [
        self.model_1,
        self.model_2,
        self.model_3,
        self.model_4,
        self.model_5,
    ]

    # Each Model will train on a rolling window of 2 Examples.
    for i, model in enumerate(self.models):
      self.train_on_examples(model, self.examples[i : i + 2], transform_graph)

  def testSpanDrivenEvaluatorInputs_InvalidArguments_RaisesInvalidArgument(
      self,
  ):
    with self.assertRaises(exceptions.InvalidArgument):
      self._span_driven_evaluator(
          self.models, self.examples, wait_spans_before_eval=-1
      )
      self._span_driven_evaluator(
          self.models, self.examples, additional_spans_per_eval=-1
      )
      self._span_driven_evaluator(
          self.models, self.examples, start_span_number=-1
      )

    try:
      self._span_driven_evaluator(
          self.models, self.examples, evaluation_training_offset=-1
      )
    except exceptions.InvalidArgument:
      self.fail(
          'A evaluation_training_offset is a valid for '
          'SpanDrivenEvaluatorInputs and exceptions.InvalidArgument should not '
          'be raised.'
      )

  def testSpanDrivenEvaluatorInputs_InvalidInputDict_RaisesInvalidArgument(
      self,
  ):
    with self.assertRaises(exceptions.InvalidArgument):
      # Invalid artifact types.
      input_dict = {
          ops_utils.MODEL_KEY: self.examples,
          ops_utils.EXAMPLES_KEY: self.models,
      }
      self._run_span_driven_evaluator(input_dict)

      # Invalid dict keys.
      input_dict = {
          ops_utils.EXAMPLES_KEY: self.examples,
          ops_utils.MODEL_KEY: self.models,
          ops_utils.MODEL_PUSH_KEY: self.models,
      }
      self._run_span_driven_evaluator(input_dict)

      # Invalid and missing dict keys.
      input_dict = {
          ops_utils.EXAMPLES_KEY: self.examples,
          ops_utils.MODEL_PUSH_KEY: self.models,
      }
      self._run_span_driven_evaluator(input_dict)

      # Input dict only contains "model" and is missing "examples" key.
      input_dict = {ops_utils.MODEL_KEY: self.models}
      self._run_span_driven_evaluator(input_dict)

  def testSpanDrivenEvaluatorInputs_EmptyInputDict_RaisesSkipSignal(self):
    with self.assertRaises(exceptions.SkipSignal):
      # Empty input_dict.
      self._run_span_driven_evaluator({})

      # "models" and "examples" keys present but do not contain artifacts.
      self._span_driven_evaluator([], [])
      self._span_driven_evaluator(self.models, [])
      self._span_driven_evaluator([], self.examples)

  def testSpanDrivenEvaluatorInputs_NoArguments(self):
    actual = self._span_driven_evaluator(
        self.models,
        self.examples[:9],
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_9],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples[:8],
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_8],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples[:3],
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_1],
        ops_utils.EXAMPLES_KEY: [self.examples_3],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models[:3],
        self.examples,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_3],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    with self.assertRaises(exceptions.SkipSignal):
      actual = self._span_driven_evaluator(
          [self.model_1],
          [self.examples_1, self.examples_2],
      )

      actual = self._span_driven_evaluator(
          self.models,
          self.examples[:5],
      )

  def testSpanDrivenEvaluatorInputs_WaitSpansBeforeEval(self):
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        wait_spans_before_eval=0,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        wait_spans_before_eval=1,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_9],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        wait_spans_before_eval=2,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_8],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        wait_spans_before_eval=3,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_7],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # end_span will be 10 - 4 = 6, Model 4 is the latest model not trained on
    # span 6.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        wait_spans_before_eval=4,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_4],
        ops_utils.EXAMPLES_KEY: [self.examples_6],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # end_span will be 10 - 5 = 5, Model 3 is the latest model not trained on
    # span 5.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        wait_spans_before_eval=5,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_3],
        ops_utils.EXAMPLES_KEY: [self.examples_5],
    }
    self.assertArtifactMapsEqual(actual, expected)

    with self.assertRaises(exceptions.SkipSignal):
      # end_span will be 10 - 8 = 2 and no model was trained on a span < 2.
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          wait_spans_before_eval=8,
      )

      # end_span will be < 1
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          wait_spans_before_eval=50,
      )

  def testSpanDrivenEvaluatorInputs_AdditionalSpansPerEval(self):
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        additional_spans_per_eval=0,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        additional_spans_per_eval=1,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_9, self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        additional_spans_per_eval=2,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [
            self.examples_8,
            self.examples_9,
            self.examples_10,
        ],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # start_span will be 10 - 4 = 6 and Model 4 is the latest model not trained
    # on span 6.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        additional_spans_per_eval=4,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_4],
        ops_utils.EXAMPLES_KEY: [
            self.examples_6,
            self.examples_7,
            self.examples_8,
            self.examples_9,
            self.examples_10,
        ],
    }
    self.assertArtifactMapsEqual(actual, expected)

    with self.assertRaises(exceptions.SkipSignal):
      # start_span will be 10 - 8 = 2 and no model was trained on a span < 2.
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          additional_spans_per_eval=8,
      )

      # start_span will be < 1
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          additional_spans_per_eval=50,
      )

  def testSpanDrivenEvaluatorInputs_EvaluationTrainingOffset(self):
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        evaluation_training_offset=0,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        evaluation_training_offset=1,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # max_span for _get_model_to_evaluate will be 10 - 4 = 6 and Model 4 is the
    # latest model not trained on span 6.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        evaluation_training_offset=4,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_4],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        evaluation_training_offset=-1,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # max_span for _get_model_to_evaluate will be 6 - (-1) = 7 and Model 5 is
    # latest model not trained on span 7.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples[:6],
        evaluation_training_offset=-1,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_6],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # max_span for _get_model_to_evaluate will be 3 - (-2) = 5 and Model 3 is
    # latest model not trained on span 5.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples[:3],
        evaluation_training_offset=-2,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_3],
        ops_utils.EXAMPLES_KEY: [self.examples_3],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # max_span for _get_model_to_evaluate will be 10 - (-50) = 55 and Model 5 is
    # latest model not trained on span 55.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        evaluation_training_offset=-50,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    with self.assertRaises(exceptions.SkipSignal):
      # max_span will be 10 - 8 = 2 and no model was trained on a span < 2.
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          evaluation_training_offset=8,
      )

      # max_span will be < 1
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          evaluation_training_offset=50,
      )

  def testSpanDrivenEvaluatorInputs_StartSpanNumber(self):
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        start_span_number=0,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        start_span_number=1,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        start_span_number=5,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [self.examples_10],
    }
    self.assertArtifactMapsEqual(actual, expected)

    with self.assertRaises(exceptions.SkipSignal):
      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          start_span_number=11,
      )

      actual = self._span_driven_evaluator(
          self.models,
          self.examples,
          start_span_number=50,
      )

  def testSpanDrivenEvaluatorInputs_MultipleVersions(self):
    examples_3_2 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 3, 'version': 2}
    )

    examples_3_3 = self.prepare_tfx_artifact(
        test_utils.Examples, properties={'span': 3, 'version': 3}
    )

    actual = self._span_driven_evaluator(
        self.models,
        [self.examples_3, examples_3_2, examples_3_3],
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_1],
        # Only the latest version of Examples with span 3 should be considered.
        ops_utils.EXAMPLES_KEY: [examples_3_3],
    }
    self.assertArtifactMapsEqual(actual, expected)

  def testSpanDrivenEvaluatorInputs_AllArguments(self):
    # Evaluates the model on the next 3 day rolling window.
    # [start_span, end_span] is [9 - 1 - 3, 9 - 1] = [5, 8]. max_span is
    # 5 - 2 = 3. Model 1 is the latest model that has not trained on span 3.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples[:9],
        wait_spans_before_eval=1,
        evaluation_training_offset=2,
        additional_spans_per_eval=3,
    )
    expected = {
        ops_utils.MODEL_KEY: [
            self.model_1
        ],  # Model 1 has only trained on spans 1 and 2.
        ops_utils.EXAMPLES_KEY: [
            self.examples_5,
            self.examples_6,
            self.examples_7,
            self.examples_8,
        ],
    }
    self.assertArtifactMapsEqual(actual, expected)

    # Evaluates the model on last-1-week rolling window. [start_span, end_span]
    # is [10 - 6, 10] = [4, 10]. max_span is 4 - (-7) = 11. Model 5 is the
    # latest model that has not trained on span 11.
    actual = self._span_driven_evaluator(
        self.models,
        self.examples,
        additional_spans_per_eval=6,
        evaluation_training_offset=-7,
    )
    expected = {
        ops_utils.MODEL_KEY: [self.model_5],
        ops_utils.EXAMPLES_KEY: [
            self.examples_4,
            self.examples_5,
            self.examples_6,
            self.examples_7,
            self.examples_8,
            self.examples_9,
            self.examples_10,
        ],
    }
    self.assertArtifactMapsEqual(actual, expected)


if __name__ == '__main__':
  tf.test.main()
