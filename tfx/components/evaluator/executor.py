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

from absl import logging
import apache_beam as beam
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import constants as tfma_constants
# Need to import the following module so that the fairness indicator post-export
# metric is registered.
import tensorflow_model_analysis.addons.fairness.post_export_metrics.fairness_indicators  # pylint: disable=unused-import
from tfx import types
from tfx.components.base import base_executor
from tfx.components.evaluator import constants
from tfx.components.util import tfxio_utils
from tfx.components.util import udf_utils
from tfx.proto import evaluator_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import path_utils
from tfx_bsl.tfxio import tensor_adapter

from google.protobuf import json_format


_TELEMETRY_DESCRIPTORS = ['Evaluator']
# TODO(pachristopher): After TFMA is released, make TFXIO as the default path.
_USE_TFXIO = False


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
        - example_splits: JSON-serialized list of names of splits on which the
          metrics are computed. Default behavior (when example_splits is set to
          None) is using the 'eval' split.

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
      raise ValueError('There can be only one candidate model, there are %d.' %
                       (len(input_dict[constants.MODEL_KEY])))
    if constants.BASELINE_MODEL_KEY in input_dict and len(
        input_dict[constants.BASELINE_MODEL_KEY]) > 1:
      raise ValueError('There can be only one baseline model, there are %d.' %
                       (len(input_dict[constants.BASELINE_MODEL_KEY])))

    self._log_startup(input_dict, output_dict, exec_properties)

    # Add fairness indicator metric callback if necessary.
    fairness_indicator_thresholds = exec_properties.get(
        'fairness_indicator_thresholds', None)
    add_metrics_callbacks = None
    if fairness_indicator_thresholds:
      add_metrics_callbacks = [
          tfma.post_export_metrics.fairness_indicators(  # pytype: disable=module-attr
              thresholds=fairness_indicator_thresholds),
      ]

    output_uri = artifact_utils.get_single_uri(
        output_dict[constants.EVALUATION_KEY])

    eval_shared_model_fn = udf_utils.try_get_fn(
        exec_properties=exec_properties,
        fn_name='custom_eval_shared_model') or tfma.default_eval_shared_model

    run_validation = False
    models = []
    if 'eval_config' in exec_properties and exec_properties['eval_config']:
      slice_spec = None
      has_baseline = bool(input_dict.get(constants.BASELINE_MODEL_KEY))
      eval_config = tfma.EvalConfig()
      json_format.Parse(exec_properties['eval_config'], eval_config)
      eval_config = tfma.update_eval_config_with_defaults(
          eval_config,
          maybe_add_baseline=has_baseline,
          maybe_remove_baseline=not has_baseline)
      tfma.verify_eval_config(eval_config)
      # Do not validate model when there is no thresholds configured. This is to
      # avoid accidentally blessing models when users forget to set thresholds.
      run_validation = bool(tfma.metrics.metric_thresholds_from_metrics_specs(
          eval_config.metrics_specs))
      if len(eval_config.model_specs) > 2:
        raise ValueError(
            """Cannot support more than two models. There are %d models in this
             eval_config.""" % (len(eval_config.model_specs)))
      # Extract model artifacts.
      for model_spec in eval_config.model_specs:
        if model_spec.is_baseline:
          model_uri = artifact_utils.get_single_uri(
              input_dict[constants.BASELINE_MODEL_KEY])
        else:
          model_uri = artifact_utils.get_single_uri(
              input_dict[constants.MODEL_KEY])
        if tfma.get_model_type(model_spec) == tfma.TF_ESTIMATOR:
          model_path = path_utils.eval_model_path(model_uri)
        else:
          model_path = path_utils.serving_model_path(model_uri)
        logging.info('Using %s as %s model.', model_path, model_spec.name)
        models.append(
            eval_shared_model_fn(
                eval_saved_model_path=model_path,
                model_name=model_spec.name,
                eval_config=eval_config,
                add_metrics_callbacks=add_metrics_callbacks))
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
      model_uri = artifact_utils.get_single_uri(input_dict[constants.MODEL_KEY])
      model_path = path_utils.eval_model_path(model_uri)
      logging.info('Using %s for model eval.', model_path)
      models.append(
          eval_shared_model_fn(
              eval_saved_model_path=model_path,
              model_name='',
              eval_config=None,
              add_metrics_callbacks=add_metrics_callbacks))

    eval_shared_model = models[0] if len(models) == 1 else models
    schema = None
    if constants.SCHEMA_KEY in input_dict:
      schema = io_utils.SchemaReader().read(
          io_utils.get_only_uri_in_dir(
              artifact_utils.get_single_uri(input_dict[constants.SCHEMA_KEY])))

    # Load and deserialize example splits from execution properties.
    example_splits = json_utils.loads(
        exec_properties.get(constants.EXAMPLE_SPLITS_KEY, 'null'))
    if not example_splits:
      example_splits = ['eval']
      logging.info("The 'example_splits' parameter is not set, using 'eval' "
                   'split.')

    logging.info('Evaluating model.')
    with self._make_beam_pipeline() as pipeline:
      examples_list = []
      tensor_adapter_config = None
      # pylint: disable=expression-not-assigned
      if _USE_TFXIO and tfma.is_batched_input(eval_shared_model, eval_config):
        tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
            examples=[
                artifact_utils.get_single_instance(
                    input_dict[constants.EXAMPLES_KEY])
            ],
            telemetry_descriptors=_TELEMETRY_DESCRIPTORS,
            schema=schema,
            raw_record_column_name=tfma_constants.ARROW_INPUT_COLUMN)
        # TODO(b/161935932): refactor after TFXIO supports multiple patterns.
        for split in example_splits:
          file_pattern = io_utils.all_files_pattern(
              artifact_utils.get_split_uri(input_dict[constants.EXAMPLES_KEY],
                                           split))
          tfxio = tfxio_factory(file_pattern)
          data = (
              pipeline
              | 'ReadFromTFRecordToArrow[%s]' % split >> tfxio.BeamSource())
          examples_list.append(data)
        if schema is not None:
          # Use last tfxio as TensorRepresentations and ArrowSchema are fixed.
          tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
              arrow_schema=tfxio.ArrowSchema(),
              tensor_representations=tfxio.TensorRepresentations())
      else:
        for split in example_splits:
          file_pattern = io_utils.all_files_pattern(
              artifact_utils.get_split_uri(input_dict[constants.EXAMPLES_KEY],
                                           split))
          data = (
              pipeline
              | 'ReadFromTFRecord[%s]' % split >>
              beam.io.ReadFromTFRecord(file_pattern=file_pattern))
          examples_list.append(data)

      custom_extractors = udf_utils.try_get_fn(
          exec_properties=exec_properties, fn_name='custom_extractors')
      extractors = None
      if custom_extractors:
        extractors = custom_extractors(
            eval_shared_model=eval_shared_model,
            eval_config=eval_config,
            tensor_adapter_config=tensor_adapter_config)

      (examples_list | 'FlattenExamples' >> beam.Flatten()
       |
       'ExtractEvaluateAndWriteResults' >> tfma.ExtractEvaluateAndWriteResults(
           eval_shared_model=models[0] if len(models) == 1 else models,
           eval_config=eval_config,
           extractors=extractors,
           output_path=output_uri,
           slice_spec=slice_spec,
           tensor_adapter_config=tensor_adapter_config))
    logging.info('Evaluation complete. Results written to %s.', output_uri)

    if not run_validation:
      # TODO(jinhuang): delete the BLESSING_KEY from output_dict when supported.
      logging.info('No threshold configured, will not validate model.')
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
    if input_dict.get(constants.BASELINE_MODEL_KEY):
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
    logging.info('Checking validation results.')
    validation_result = tfma.load_validation_result(output_uri)
    if validation_result.validation_ok:
      io_utils.write_string_file(
          os.path.join(blessing.uri, constants.BLESSED_FILE_NAME), '')
      blessing.set_int_custom_property(constants.ARTIFACT_PROPERTY_BLESSED_KEY,
                                       constants.BLESSED_VALUE)
    else:
      io_utils.write_string_file(
          os.path.join(blessing.uri, constants.NOT_BLESSED_FILE_NAME), '')
      blessing.set_int_custom_property(constants.ARTIFACT_PROPERTY_BLESSED_KEY,
                                       constants.NOT_BLESSED_VALUE)
    logging.info('Blessing result %s written to %s.',
                 validation_result.validation_ok, blessing.uri)
