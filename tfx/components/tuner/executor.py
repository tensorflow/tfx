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

import json
import os
from typing import Any, Callable, Dict, List, Optional

from absl import logging
from keras_tuner.engine import base_tuner
from tfx import types
from tfx.components.trainer import fn_args_utils
from tfx.components.util import udf_utils
from tfx.dsl.components.base import base_executor
from tfx.proto import tuner_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import proto_utils


# Default file name for generated best hyperparameters file.
_DEFAULT_FILE_NAME = 'best_hyperparameters.txt'


# TODO(b/160253334): Establish a separation of practice between this 'default'
#                    module and the ones in 'extensions'.
def _get_tuner_fn(exec_properties: Dict[str, Any]) -> Callable[..., Any]:
  """Returns tuner_fn from execution properties."""
  return udf_utils.get_fn(exec_properties, 'tuner_fn')


def get_tune_args(
    exec_properties: Dict[str, Any]) -> Optional[tuner_pb2.TuneArgs]:
  """Returns TuneArgs protos from execution properties, if present."""
  tune_args = exec_properties.get(standard_component_specs.TUNE_ARGS_KEY)
  if not tune_args:
    return None

  result = tuner_pb2.TuneArgs()
  proto_utils.json_to_proto(tune_args, result)

  return result


def write_best_hyperparameters(
    tuner: base_tuner.BaseTuner,
    output_dict: Dict[str, List[types.Artifact]]) -> None:
  """Write out best hyperpeameters known to the given Tuner instance."""
  best_hparams_config = tuner.get_best_hyperparameters()[0].get_config()
  logging.info('Best HyperParameters: %s', best_hparams_config)
  best_hparams_path = os.path.join(
      artifact_utils.get_single_uri(
          output_dict[standard_component_specs.BEST_HYPERPARAMETERS_KEY]),
      _DEFAULT_FILE_NAME)
  io_utils.write_string_file(best_hparams_path, json.dumps(best_hparams_config))
  logging.info('Best Hyperparameters are written to %s.', best_hparams_path)


def search(input_dict: Dict[str, List[types.Artifact]],
           exec_properties: Dict[str, Any],
           working_dir: str) -> base_tuner.BaseTuner:
  """Conduct a single hyperparameter search loop, and return the Tuner."""
  tuner_fn = _get_tuner_fn(exec_properties)

  fn_args = fn_args_utils.get_common_fn_args(input_dict, exec_properties,
                                             working_dir)

  tuner_fn_result = tuner_fn(fn_args)
  result = tuner_fn_result.tuner

  # TODO(b/156966497): set logger for printing.
  result.search_space_summary()
  logging.info('Start tuning... Tuner ID: %s', result.tuner_id)
  result.search(**tuner_fn_result.fit_kwargs)
  logging.info('Finished tuning... Tuner ID: %s', result.tuner_id)
  result.results_summary()

  return result


class Executor(base_executor.BaseExecutor):
  """TFX Tuner component executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    if get_tune_args(exec_properties):
      raise ValueError("TuneArgs is not supported by this Tuner's Executor.")

    tuner = search(input_dict, exec_properties, self._get_tmp_dir())

    write_best_hyperparameters(tuner, output_dict)
