# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for SpanDrivenEvaluatorInputs operator."""

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.dsl.input_resolution.ops import training_range_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.utils import typing_utils


def _validate_input_dict(input_dict: typing_utils.ArtifactMultiMap):
  """Checks that the input_dict is properly formatted."""
  if not input_dict:
    raise exceptions.SkipSignal()

  valid_keys = {ops_utils.MODEL_KEY, ops_utils.EXAMPLES_KEY}
  ops_utils.validate_input_dict(input_dict, valid_keys, requires_all=True)

  if (
      not input_dict[ops_utils.MODEL_KEY]
      or not input_dict[ops_utils.EXAMPLES_KEY]
  ):
    raise exceptions.SkipSignal()


class SpanDrivenEvaluatorInputs(
    resolver_op.ResolverOp,
    canonical_name='tfx.SpanDrivenEvaluatorInputs',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """SpanDrivenEvaluatorInputs operator."""

  # TODO(b/265378460): Rename these parameters and update comments to be more
  # user friendly.

  # If 0, as soon as a new example span appears, we perform an evaluation
  # on it.  Otherwise, when example span (D + wait_spans_before_eval) first
  # appears, we perform an evaluation on example span D.
  wait_spans_before_eval = resolver_op.Property(type=int, default=0)

  # Indicates how to choose the model checkpoint for evaluation.
  # When evaluating examples from span D, we choose the latest model
  # checkpoint which did not train on span (D - evaluation_training_offset).
  # Note that negative values are allowed here intentionally.
  evaluation_training_offset = resolver_op.Property(type=int, default=0)

  # Indicates how many spans should be included with an evaluation at span
  # D. If 0, then only span D's Examples are evaluated.  Otherwise when
  # evaluating using span D, we will also include examples in D - 1, D - 2, ...
  # up through D - additional_spans_per_eval.
  additional_spans_per_eval = resolver_op.Property(type=int, default=0)

  # Indicates the smallest span number to be considered. Only spans
  # >= start_span_number will be considered.
  start_span_number = resolver_op.Property(type=int, default=0)

  def _get_model_to_evaluate(
      self,
      trained_examples_by_model: Dict[types.Artifact, List[types.Artifact]],
      max_span: int,
  ) -> Optional[types.Artifact]:
    """Finds the latest Model not trained on Examples with span max_span."""
    # Recall that the trained_examples_by_model keys are already sorted from
    # latest created to oldest created.
    for model, trained_examples in trained_examples_by_model.items():
      # If no examples were returned (e.g. they were garbage collected), then
      # continue searching.
      if not trained_examples:
        continue

      if trained_examples[-1].span < max_span:
        # The first Model was trained on spans less than max_span.
        return model

    # No elegible Model was found, so a SkipSignal is raised.
    raise exceptions.SkipSignal()

  def apply(self, input_dict: typing_utils.ArtifactMultiDict):
    """Returns a single Model and the list of Examples to evaluate it with.

    The input_dict is expected to have the following format:

    {
        "model": [Model 1, Model 2, ...],
        "examples": [Examples 1, Examples 2...]
    }

    where "model" and "examples" are required keys.

    Only the latest version of each Example is considered. Also note that only
    the standard TFleX Model and Examples artifacts are supported.

    Args:
      input_dict: An input dict containing "model" and "examples" as keys and
        lists of Model and Examples, respectively.

    Returns:
      A dictionary containing a single Model and the list of Examples to
      evaluate it with. Note that the Model will be latest created, eligible
      model.

      For example:
      {
          "model": [Model 3],
          "examples": [Examples 3, Examples 4]
      }

    Raises:
      InvalidArgument: If the input_dict is malformed.
      SkipSignal:
        1) input_dict is empty or either of its values are empty.
        2) The evaluation span range contains spans not present in the passed in
           Examples.
        3) No Model was found that was trained on an Example with a small enough
           span.
    """
    _validate_input_dict(input_dict)

    ops_utils.validate_argument(
        'wait_spans_before_eval', self.wait_spans_before_eval, min_value=0
    )
    ops_utils.validate_argument(
        'start_span_number', self.start_span_number, min_value=0
    )
    ops_utils.validate_argument(
        'additional_spans_per_eval', self.additional_spans_per_eval, min_value=0
    )

    # Precompute the training Examples for each Model.
    trained_examples_by_model = {}
    for model in input_dict[ops_utils.MODEL_KEY]:
      trained_examples_by_model[model] = training_range_op.training_range(
          self.context.store, model
      )

    # Sort the Models by latest created, with ties broken by id.
    trained_examples_by_model = dict(
        sorted(
            trained_examples_by_model.items(),
            key=lambda kv: (  # pylint: disable=g-long-lambda
                kv[0].mlmd_artifact.create_time_since_epoch,
                kv[0].id,
            ),
            reverse=True,
        )
    )

    # Sort Examples by span, then version, ties broken by id.
    sorted_examples = ops_utils.filter_artifacts_by_span(
        artifacts=input_dict[ops_utils.EXAMPLES_KEY],
        span_descending=False,
        n=0,
        min_span=self.start_span_number,
        keep_all_versions=False,
    )
    if not sorted_examples:
      raise exceptions.SkipSignal()
    min_span = sorted_examples[0].span
    max_span = sorted_examples[-1].span

    # Consider Examples in the range (inclusive):
    # [max_span - wait_spans_before_eval - additional_spans_per_eval,
    #  max_span - wait_spans_before_eval]
    end_span = max_span - self.wait_spans_before_eval
    start_span = end_span - self.additional_spans_per_eval

    if start_span < min_span or end_span < min_span:
      raise exceptions.SkipSignal()

    # Find the Model trained on Examples with a span less than
    # start_span - self.evaluation_training_offset.
    model = self._get_model_to_evaluate(
        trained_examples_by_model,
        start_span - self.evaluation_training_offset,
    )

    # Keep Examples that fall in [start_span, end_span], inclusive.
    examples_window = [
        e
        for e in sorted_examples
        if e.span >= start_span and e.span <= end_span
    ]

    # The evaluation dictionary will contain the latest created and eligible
    # Model to evaluate, and the Examples to evaluate it with.
    return {
        ops_utils.MODEL_KEY: [model],
        ops_utils.EXAMPLES_KEY: examples_window,
    }
