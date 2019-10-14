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
"""Generic TFX model evaluator executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import apache_beam as beam
import tensorflow_model_analysis as tfma
from typing import Any, Dict, List, Text

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.proto import evaluator_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils


class Executor(base_executor.BaseExecutor):
  """Generic TFX model evaluator executor."""

  def _get_slice_spec_from_feature_slicing_spec(
      self, spec: evaluator_pb2.FeatureSlicingSpec
  ) -> List[tfma.slicer.SingleSliceSpec]:
    """Given a feature slicing spec, returns a List of SingleSliceSpecs.

    Args:
      spec: slice specification.

    Returns:
      List of corresponding SingleSliceSpecs. Always includes the overall slice,
      even if it was not specified in the given spec.
    """
    result = []
    for single_spec in spec.specs:
      columns = single_spec.column_for_slicing
      result.append(tfma.slicer.SingleSliceSpec(columns=columns))
    # Always include the overall slice.
    if tfma.slicer.SingleSliceSpec() not in result:
      result.append(tfma.slicer.SingleSliceSpec())
    return result

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Runs a batch job to evaluate the eval_model against the given input.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - model_exports: exported model.
        - examples: examples for eval the model.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: model evaluation results.
      exec_properties: A dict of execution properties.
        - feature_slicing_spec: JSON string of evaluator_pb2.FeatureSlicingSpec
          instance, providing the way to slice the data.

    Returns:
      None
    """
    if 'model_exports' not in input_dict:
      raise ValueError('\'model_exports\' is missing in input dict.')
    if 'examples' not in input_dict:
      raise ValueError('\'examples\' is missing in input dict.')
    if 'output' not in output_dict:
      raise ValueError('\'output\' is missing in output dict.')

    self._log_startup(input_dict, output_dict, exec_properties)

    # Extract input artifacts
    model_exports_uri = artifact_utils.get_single_uri(
        input_dict['model_exports'])

    feature_slicing_spec = evaluator_pb2.FeatureSlicingSpec()
    json_format.Parse(exec_properties['feature_slicing_spec'],
                      feature_slicing_spec)
    slice_spec = self._get_slice_spec_from_feature_slicing_spec(
        feature_slicing_spec)

    output_uri = artifact_utils.get_single_uri(output_dict['output'])

    eval_model_path = path_utils.eval_model_path(model_exports_uri)

    # Add fairness indicator metric callback if necessary.
    fairness_indicator_thresholds = exec_properties.get(
        'fairness_indicator_thresholds', None)
    add_metrics_callbacks = None
    if fairness_indicator_thresholds:
      # Need to import the following module so that the fairness indicator
      # post-export metric is registered.
      import tensorflow_model_analysis.addons.fairness.post_export_metrics.fairness_indicators  # pylint: disable=g-import-not-at-top, unused-variable
      add_metrics_callbacks = [
          tfma.post_export_metrics.fairness_indicators(  # pytype: disable=module-attr
              thresholds=fairness_indicator_thresholds),
      ]

    absl.logging.info('Using {} for model eval.'.format(eval_model_path))
    eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=eval_model_path,
        add_metrics_callbacks=add_metrics_callbacks)

    absl.logging.info('Evaluating model.')
    with self._make_beam_pipeline() as pipeline:
      # pylint: disable=expression-not-assigned
      (pipeline
       | 'ReadData' >> beam.io.ReadFromTFRecord(
           file_pattern=io_utils.all_files_pattern(
               artifact_utils.get_split_uri(input_dict['examples'], 'eval')))
       |
       'ExtractEvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
           eval_shared_model=eval_shared_model,
           slice_spec=slice_spec,
           output_path=output_uri))
    absl.logging.info(
        'Evaluation complete. Results written to {}.'.format(output_uri))
