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
"""Generic TFX tuner executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from typing import Any, Dict, List, Text
import absl
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import import_utils
from tfx.utils import io_utils

# Default file name for generated best hyperparameters file.
_DEFAULT_FILE_NAME = 'best_hyperparameters.txt'


class Executor(base_executor.BaseExecutor):
  """TFX Tuner component executor."""

  def _GetTunerFn(self, exec_properties: Dict[Text, Any]) -> Any:
    """Loads and returns user-defined tuner_fn."""

    has_module_file = bool(exec_properties.get('module_file'))
    has_tuner_fn = bool(exec_properties.get('tuner_fn'))

    if has_module_file == has_tuner_fn:
      raise ValueError(
          "Neither or both of 'module_file' 'tuner_fn' have been supplied in "
          "'exec_properties'.")

    if has_module_file:
      return import_utils.import_func_from_source(
          exec_properties['module_file'], 'tuner_fn')

    tuner_fn_path_split = exec_properties['tuner_fn'].split('.')
    return import_utils.import_func_from_module(
        '.'.join(tuner_fn_path_split[0:-1]), tuner_fn_path_split[-1])

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    # KerasTuner generates tuning state (e.g., oracle, trials) to working dir.
    working_dir = self._get_tmp_dir()

    train_path = artifact_utils.get_split_uri(input_dict['examples'], 'train')
    eval_path = artifact_utils.get_split_uri(input_dict['examples'], 'eval')
    schema_file = io_utils.get_only_uri_in_dir(
        artifact_utils.get_single_uri(input_dict['schema']))
    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())

    tuner_fn = self._GetTunerFn(exec_properties)
    tuner_spec = tuner_fn(working_dir, io_utils.all_files_pattern(train_path),
                          io_utils.all_files_pattern(eval_path), schema)
    tuner = tuner_spec.tuner

    tuner.search_space_summary()
    # TODO(jyzhao): assert v2 behavior as KerasTuner doesn't work in v1.
    # TODO(jyzhao): make steps configurable or move search() to module file.
    tuner.search(
        tuner_spec.train_dataset,
        steps_per_epoch=1000,
        validation_steps=500,
        validation_data=tuner_spec.eval_dataset)
    tuner.results_summary()

    best_hparams = tuner.oracle.get_best_trials(
        1)[0].hyperparameters.get_config()
    best_hparams_path = os.path.join(
        artifact_utils.get_single_uri(output_dict['best_hyperparameters']),
        _DEFAULT_FILE_NAME)
    io_utils.write_string_file(best_hparams_path, json.dumps(best_hparams))
    absl.logging.info('Best HParams is written to %s.' % best_hparams_path)

    # TODO(jyzhao): export best tuning model.
