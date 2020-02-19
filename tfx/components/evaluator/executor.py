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
"""Generic TFX model evaluator executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

import absl
import apache_beam as beam
import tensorflow as tf
import tensorflow_model_analysis as tfma

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.evaluator import constants
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
        - eval_config: JSON string of tfma.EvalConfig.
        - feature_slicing_spec: JSON string of evaluator_pb2.FeatureSlicingSpec
          instance, providing the way to slice the data. Deprecated, use
          eval_config.slicing_specs instead.

    Returns:
      None
    """
    if constants.EXAMPLES_KEY not in input_dict:
      raise ValueError('EXAMPLES_KEY is missing from input dict.')
    if constants.MODEL_KEY not in input_dict:
      raise ValueError('MODEL_KEY is missing from input dict.')
    if constants.EVALUATION_KEY not in output_dict:
      raise ValueError('EVALUATION_KEY is missing from output dict.')
    if len(input_dict[constants.MODEL_KEY]) > 1:
      raise ValueError(
          'There can be only one candidate model, there are {}.'.format(
              len(input_dict[constants.MODEL_KEY])))
    if constants.BASELINE_MODEL_KEY in input_dict and len(
        input_dict[constants.BASELINE_MODEL_KEY]) > 1:
      raise ValueError(
          'There can be only one baseline model, there are {}.'.format(
              len(input_dict[constants.BASELINE_MODEL_KEY])))

    self._log_startup(input_dict, output_dict, exec_properties)

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

    def _get_eval_saved_model(artifact: List[types.Artifact],
                              tags=None) -> tfma.EvalSharedModel:
      model_uri = artifact_utils.get_single_uri(artifact)
      if tags and tf.saved_model.SERVING in tags:
        model_path = path_utils.serving_model_path(model_uri)
      else:
        model_path = path_utils.eval_model_path(model_uri)
      return tfma.default_eval_shared_model(
          eval_saved_model_path=model_path,
          tags=tags,
          add_metrics_callbacks=add_metrics_callbacks)

    output_uri = artifact_utils.get_single_uri(
        output_dict[constants.EVALUATION_KEY])

    run_validation = False
    if 'eval_config' in exec_properties and exec_properties['eval_config']:
      slice_spec = None
      eval_config = tfma.EvalConfig()
      json_format.Parse(exec_properties['eval_config'], eval_config)
      # Do not validate model when there is no thresholds configured. This is to
      # avoid accidentally blessing models when users forget to set thresholds.
      for metrics_spec in eval_config.metrics_specs:
        if (metrics_spec.thresholds or any(
            metric.HasField('threshold') for metric in metrics_spec.metrics)):
          run_validation = True
          break
      # Extract model artifacts.
      # Baseline will be ignored if baseline is not configured in model_spec.
      if len(eval_config.model_specs) > 2:
        raise ValueError(
            """Cannot support more than two models. There are {} models in this
             eval_config.""".format(len(eval_config.model_specs)))
      models = {}
      if not eval_config.model_specs:
        eval_config.model_specs.add()
      for model_spec in eval_config.model_specs:
        if model_spec.signature_name != 'eval':
          tags = [tf.saved_model.SERVING]
        if model_spec.is_baseline:
          if constants.BASELINE_MODEL_KEY not in input_dict:
            absl.logging.info("""No baseline model provided, ignoring all
                change thresholds.""")
            for metrics_spec in eval_config.metrics_specs:
              for metric in metrics_spec.metrics:
                metric.threshold.clear_change_threshold()
              for threshold in metrics_spec.thresholds.values():
                threshold.clear_change_threshold()
          models[model_spec.name] = _get_eval_saved_model(
              input_dict[constants.BASELINE_MODEL_KEY], tags)
          absl.logging.info('Using {} as baseline model.'.format(
              models[model_spec.name].model_path))
        else:
          models[model_spec.name] = _get_eval_saved_model(
              input_dict[constants.MODEL_KEY], tags)
          absl.logging.info('Using {} for model eval.'.format(
              models[model_spec.name].model_path))
    else:
      eval_config = None
      assert ('feature_slicing_spec' in exec_properties and
              exec_properties['feature_slicing_spec']
             ), 'both eval_config and feature_slicing_spec are unset.'
      feature_slicing_spec = evaluator_pb2.FeatureSlicingSpec()
      json_format.Parse(exec_properties['feature_slicing_spec'],
                        feature_slicing_spec)
      slice_spec = self._get_slice_spec_from_feature_slicing_spec(
          feature_slicing_spec)
      models = _get_eval_saved_model(input_dict[constants.MODEL_KEY])
      absl.logging.info('Using {} for model eval.'.format(models.model_path))

    absl.logging.info('Evaluating model.')
    with self._make_beam_pipeline() as pipeline:
      # pylint: disable=expression-not-assigned
      (pipeline
       | 'ReadData' >> beam.io.ReadFromTFRecord(
           file_pattern=io_utils.all_files_pattern(
               artifact_utils.get_split_uri(input_dict[constants.EXAMPLES_KEY],
                                            'eval')))
       |
       'ExtractEvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
           eval_shared_model=models,
           eval_config=eval_config,
           output_path=output_uri,
           slice_spec=slice_spec))
    absl.logging.info(
        'Evaluation complete. Results written to {}.'.format(output_uri))

    if not run_validation:
      # TODO(jinhuang): delete the BLESSING_KEY from output_dict when supported.
      absl.logging.info('No threshold configured, will not validate model.')
      return
    # Set up blessing artifact
    blessing = artifact_utils.get_single_instance(
        output_dict[constants.BLESSING_KEY])
    blessing.set_string_custom_property(
        constants.ARTIFACT_PROPERTY_CURRENT_MODEL_URI_KEY,
        artifact_utils.get_single_uri(input_dict[constants.MODEL_KEY]))
    blessing.set_int_custom_property(
        constants.ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY,
        input_dict[constants.MODEL_KEY][0].id)
    if constants.BASELINE_MODEL_KEY in input_dict:
      baseline_model = input_dict[constants.BASELINE_MODEL_KEY][0]
      blessing.set_string_custom_property(
          constants.ARTIFACT_PROPERTY_BASELINE_MODEL_URI_KEY,
          baseline_model.uri)
      blessing.set_int_custom_property(
          constants.ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY, baseline_model.id)
    if 'current_component_id' in exec_properties:
      blessing.set_string_custom_property(
          'component_id', exec_properties['current_component_id'])
    # Check validation result and write BLESSED file accordingly.
    validation_file = os.path.join(output_uri, tfma.constants.VALIDATIONS_KEY)
    absl.logging.info('Checking validation results.')
    validation_result = tfma.load_validation_result(validation_file)
    if validation_result.validation_ok:
      io_utils.write_string_file(
          os.path.join(blessing.uri, constants.BLESSED_FILE_NAME), '')
      blessing.set_int_custom_property(constants.ARTIFACT_PROPERTY_BLESSED_KEY,
                                       constants.BLESSED_VALUE)
    absl.logging.info('Blessing result {} written to {}.'.format(
        validation_result.validation_ok, blessing.uri))
