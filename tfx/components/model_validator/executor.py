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
"""Generic TFX model validator executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import absl
import apache_beam as beam
import tensorflow as tf
import tensorflow_model_analysis as tfma
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

# Path to store model eval results for validation.
CURRENT_MODEL_EVAL_RESULT_PATH = 'eval_results/current_model/'
BLESSED_MODEL_EVAL_RESULT_PATH = 'eval_results/blessed_model/'


class Executor(base_executor.BaseExecutor):
  """TFX model validator executor.

  The model validator helps prevent bad models from being pushed to production.
  It does this by validating exported models against known good models (e.g. the
  current production model), and marking the exported model as good ("blessing
  it") only if the exported model's metrics are within predefined thresholds
  around the good model's metrics.

  The model validator will validate tf.serving format exported models produced
  by the Trainer component. The validator evaluates the models on examples
  created by the ExampleGen component. The validator will also automatically
  read data written by the Pusher component regarding the latest pushed models
  by using ml.metadata to query the previously pushed artifacts.

  To include ModelValidator in a TFX pipeline, configure your pipeline similar
  to
  https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L110.

  """

  # TODO(jyzhao): customized threshold support.
  def _pass_threshold(self, eval_result: tfma.EvalResult) -> bool:
    """Check threshold."""
    return True

  # TODO(jyzhao): customized validation support.
  def _compare_eval_result(self, current_model_eval_result: tfma.EvalResult,
                           blessed_model_eval_result: tfma.EvalResult) -> bool:
    """Compare accuracy of all metrics and return true if current is better or equal."""
    for current_metric, blessed_metric in zip(
        current_model_eval_result.slicing_metrics,
        blessed_model_eval_result.slicing_metrics):
      # slicing_metric is a tuple, index 0 is slice, index 1 is its value.
      if current_metric[0] != blessed_metric[0]:
        raise RuntimeError('EvalResult not match {} vs {}.'.format(
            current_metric[0], blessed_metric[0]))
      # TODO(b/140455644): TFMA introduced breaking change post 0.14 release.
      # Remove this forward compatibility change after 0.15 release.
      current_model_metrics = current_metric[1]
      blessed_model_metrics = blessed_metric[1]
      try:
        current_model_accuracy = current_model_metrics['accuracy']
        blessed_model_accuracy = blessed_model_metrics['accuracy']
      except KeyError:
        current_model_accuracy = current_model_metrics['']['']['accuracy']
        blessed_model_accuracy = blessed_model_metrics['']['']['accuracy']
      if (current_model_accuracy['doubleValue'] <
          blessed_model_accuracy['doubleValue']):
        absl.logging.info(
            'Current model accuracy is worse than blessed model: {}'.format(
                current_metric[0]))
        return False
    return True

  def _generate_blessing_result(self, eval_examples_uri: Text,
                                slice_spec: List[tfma.slicer.SingleSliceSpec],
                                current_model_dir: Text,
                                blessed_model_dir: Text) -> bool:
    current_model_eval_result_path = os.path.join(
        self._temp_path, CURRENT_MODEL_EVAL_RESULT_PATH)
    blessed_model_eval_result_path = os.path.join(
        self._temp_path, BLESSED_MODEL_EVAL_RESULT_PATH)

    with self._make_beam_pipeline() as pipeline:
      eval_data = (
          pipeline | 'ReadData' >> beam.io.ReadFromTFRecord(
              file_pattern=io_utils.all_files_pattern(eval_examples_uri)))

      current_model = tfma.default_eval_shared_model(
          eval_saved_model_path=path_utils.eval_model_path(current_model_dir))
      (eval_data | 'EvalCurrentModel' >> tfma.ExtractEvaluateAndWriteResults(  # pylint: disable=expression-not-assigned
          eval_shared_model=current_model,
          slice_spec=slice_spec,
          output_path=current_model_eval_result_path))

      if blessed_model_dir is not None:
        blessed_model = tfma.default_eval_shared_model(
            eval_saved_model_path=path_utils.eval_model_path(blessed_model_dir))
        (eval_data | 'EvalBlessedModel' >> tfma.ExtractEvaluateAndWriteResults(  # pylint: disable=expression-not-assigned
            eval_shared_model=blessed_model,
            slice_spec=slice_spec,
            output_path=blessed_model_eval_result_path))

    absl.logging.info('all files in current_model_eval_result_path: [%s]',
                      str(tf.io.gfile.listdir(current_model_eval_result_path)))
    current_model_eval_result = tfma.load_eval_result(
        output_path=current_model_eval_result_path)

    if not self._pass_threshold(current_model_eval_result):
      absl.logging.info('Current model does not pass threshold.')
      return False
    absl.logging.info('Current model passes threshold.')

    if blessed_model_dir is None:
      absl.logging.info('No blessed model yet.')
      return True
    absl.logging.info('all files in blessed_model_eval_result: [%s]',
                      str(tf.io.gfile.listdir(blessed_model_eval_result_path)))
    blessed_model_eval_result = tfma.load_eval_result(
        output_path=blessed_model_eval_result_path)

    if (self._compare_eval_result(current_model_eval_result,
                                  blessed_model_eval_result)):
      absl.logging.info('Current model better than blessed model.')
      return True
    else:
      absl.logging.info('Current model worse than blessed model.')
      return False

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Validate current model against last blessed model.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for eval the model.
        - model: current model for validation.
      output_dict: Output dict from output key to a list of Artifacts.
        - blessing: model blessing result.
      exec_properties: A dict of execution properties.
        - blessed_model: last blessed model for validation.
        - blessed_model_id: last blessed model id.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    self._temp_path = self._get_tmp_dir()
    absl.logging.info('Using temp path {} for tft.beam'.format(self._temp_path))

    eval_examples_uri = artifact_utils.get_split_uri(input_dict['examples'],
                                                     'eval')
    blessing = artifact_utils.get_single_instance(output_dict['blessing'])

    # Current model.
    current_model = artifact_utils.get_single_instance(input_dict['model'])
    absl.logging.info('Using {} as current model.'.format(current_model.uri))
    blessing.set_string_custom_property('current_model', current_model.uri)
    blessing.set_int_custom_property('current_model_id', current_model.id)

    # Denote model component_name.
    component_id = exec_properties['component_id']
    blessing.set_string_custom_property('component_id', component_id)

    # Blessed model.
    blessed_model_dir = exec_properties['blessed_model']
    blessed_model_id = exec_properties['blessed_model_id']
    absl.logging.info('Using {} as blessed model.'.format(blessed_model_dir))
    if blessed_model_dir:
      blessing.set_string_custom_property('blessed_model', blessed_model_dir)
      blessing.set_int_custom_property('blessed_model_id', blessed_model_id)

    absl.logging.info('Validating model.')
    # TODO(b/125853306): support customized slice spec.
    blessed = self._generate_blessing_result(
        eval_examples_uri=eval_examples_uri,
        slice_spec=[tfma.slicer.SingleSliceSpec()],
        current_model_dir=current_model.uri,
        blessed_model_dir=blessed_model_dir)

    if blessed:
      io_utils.write_string_file(os.path.join(blessing.uri, 'BLESSED'), '')
      blessing.set_int_custom_property('blessed', 1)
    else:
      io_utils.write_string_file(os.path.join(blessing.uri, 'NOT_BLESSED'), '')
      blessing.set_int_custom_property('blessed', 0)
    absl.logging.info('Blessing result {} written to {}.'.format(
        blessed, blessing.uri))

    io_utils.delete_dir(self._temp_path)
    absl.logging.info('Cleaned up temp path {} on executor success.'.format(
        self._temp_path))
