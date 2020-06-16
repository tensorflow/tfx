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
import multiprocessing
import os
from typing import Any, Dict, List, Optional, Text
import absl
import kerastuner

from tfx import types
from tfx.components.base import base_executor
from tfx.components.trainer import fn_args_utils
from tfx.components.util import udf_utils
from tfx.types import artifact_utils
from tfx.utils import io_utils

# Key for best hyperparameters in executor output_dict.
_BEST_HYPERPARAMETERS_KEY = 'best_hyperparameters'
# Key for tune args in executor exec_properties.
_TUNE_ARGS_KEY = 'tune_args'
# Default file name for generated best hyperparameters file.
_DEFAULT_FILE_NAME = 'best_hyperparameters.txt'


def _get_tuner_fn(exec_properties: Dict[Text, Any]):
  """Imports user-supplised tuner_fn at module top level."""
  return udf_utils.get_fn(exec_properties, 'tuner_fn')


def _search(exec_properties: Dict[Text, Any],
            fn_args: fn_args_utils.FnArgs,
            tuner_id: Optional[Text] = None) -> kerastuner.Tuner:
  """Multiuple tuning worker-safe tuner.search()."""
  tuner_fn = _get_tuner_fn(exec_properties)
  tuner_fn_result = tuner_fn(fn_args)

  tuner = tuner_fn_result.tuner
  if tuner_id:
    tuner.tuner_id = tuner_id

  # TODO(b/156966497): set logger for printing.
  tuner.search_space_summary()
  absl.logging.info('Start tuning (tuner_id = {})...'.format(tuner.tuner_id))

  fit_kwargs = tuner_fn_result.fit_kwargs
  tuner.search(**fit_kwargs)

  return tuner


# Defining this at top level because Beam cannot serialize lambda.
def _search_wrapper(args):
  return _search(*args)


def _maybe_concurrent_search(
    exec_properties: Dict[Text, Any],
    fn_args: fn_args_utils.FnArgs) -> List[kerastuner.Tuner]:
  """Launch tuner.search() loop possibly concurrently."""
  tune_args = exec_properties.get(_TUNE_ARGS_KEY)

  num_parallel_trials = 1
  if tune_args and tune_args.num_parallel_trials:
    num_parallel_trials = tune_args.num_parallel_trials

  p = multiprocessing.dummy.Pool(processes=num_parallel_trials)
  return p.map(_search_wrapper, [(exec_properties, fn_args, 'tuner{}'.format(i))
                                 for i in range(num_parallel_trials)])


class Executor(base_executor.BaseExecutor):
  """TFX Tuner component executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    # TODO(b/143900133): Remove this guarding when we demonstrate a Tuner with
    # Cloud backend, which handles multiple concurrent tuning loops.
    if exec_properties.get(_TUNE_ARGS_KEY):
      raise ValueError(
          "TuneArgs is not supported for default Tuner's Executor.")

    fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
                                               self._get_tmp_dir())

    # Returns Tuner instances from each searche.
    tuners = _maybe_concurrent_search(exec_properties, fn_args)

    # TODO(b/156966497): set logger for printing.
    tuners[0].results_summary()
    best_hparams_config = tuners[0].get_best_hyperparameters()[0].get_config()
    absl.logging.info('Best hyperParameters: %s' % best_hparams_config)
    best_hparams_path = os.path.join(
        artifact_utils.get_single_uri(output_dict[_BEST_HYPERPARAMETERS_KEY]),
        _DEFAULT_FILE_NAME)
    io_utils.write_string_file(best_hparams_path,
                               json.dumps(best_hparams_config))
    absl.logging.info('Best Hyperparameters are written to %s.' %
                      best_hparams_path)
