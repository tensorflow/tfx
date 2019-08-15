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
"""Base handler class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import os
import re
import subprocess
import sys
import tempfile

from six import with_metaclass
import tensorflow as tf
from typing import Any, Dict, Text, List

from tfx.tools.cli import labels
from tfx.utils import io_utils


# TODO(b/132286477): Check if _check_pipeline_folder, _get_handler_home and
# _get_handler_pipeline_path can be shifted to base_handler after all handlers
# are implemented.
class BaseHandler(with_metaclass(abc.ABCMeta, object)):
  """Base Handler for CLI.

  Attributes:
    flags_dict: A dictionary with flags provided in a command.
  """
  # TODO(b/132286477): Update comments after finalizing return types.

  def __init__(self, flags_dict: Dict[Text, Any]):
    self.flags_dict = flags_dict

  @abc.abstractmethod
  def create_pipeline(self) -> None:
    """Creates pipeline for the handler."""
    pass

  @abc.abstractmethod
  def update_pipeline(self) -> None:
    """Updates pipeline for the handler."""
    pass

  @abc.abstractmethod
  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    pass

  @abc.abstractmethod
  def delete_pipeline(self) -> None:
    """Deletes pipeline for the handler."""
    pass

  @abc.abstractmethod
  def compile_pipeline(self) -> None:
    """Compiles pipeline for the handler."""
    pass

  @abc.abstractmethod
  def create_run(self) -> None:
    """Runs a pipeline for the handler."""
    pass

  @abc.abstractmethod
  def delete_run(self) -> None:
    """Deletes a run."""
    pass

  @abc.abstractmethod
  def terminate_run(self) -> None:
    """Stops a run."""
    pass

  @abc.abstractmethod
  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    pass

  @abc.abstractmethod
  def get_run(self) -> None:
    """Checks run status."""
    pass

  def _check_pipeline_dsl_path(self) -> None:
    """Check if pipeline dsl path exists."""
    if not tf.io.gfile.exists(self.flags_dict[labels.PIPELINE_DSL_PATH]):
      sys.exit('Invalid pipeline path: {}'
               .format(self.flags_dict[labels.PIPELINE_DSL_PATH]))

  def _check_dsl_runner(self) -> None:
    """Check if runner in dsl is same as engine flag."""
    engine_flag = self.flags_dict[labels.ENGINE_FLAG]
    with open(self.flags_dict[labels.PIPELINE_DSL_PATH], 'r') as f:
      dsl_contents = f.read()
      regexes = {
          'airflow': r'AirflowDagRunner\(.*\)',
          'kubeflow': r'KubeflowDagRunner\(.*\)',
          'beam': r'BeamDagRunner\(.*\)'
      }
      match = re.search(regexes[engine_flag], dsl_contents)
      if not match:
        sys.exit('{} runner not found in dsl.'.format(engine_flag))

  def _extract_pipeline_args(self) -> Dict[Text, Any]:
    """Get pipeline args from the DSL."""
    if os.path.isdir(self.flags_dict[labels.PIPELINE_DSL_PATH]):
      sys.exit('Provide dsl file path.')

    # Create an environment for subprocess.
    temp_env = os.environ.copy()

    # Create temp file to store pipeline_args from pipeline dsl.
    temp_file = tempfile.mkstemp(prefix='cli_tmp_', suffix='_pipeline_args')[1]

    # Store temp_file path in temp_env.
    temp_env[labels.TFX_JSON_EXPORT_PIPELINE_ARGS_PATH] = temp_file

    # Run dsl with mock environment to store pipeline args in temp_file.
    self._subprocess_call(['python', self.flags_dict[labels.PIPELINE_DSL_PATH]],
                          env=temp_env)
    if os.stat(temp_file).st_size != 0:
      # Load pipeline_args from temp_file for TFX pipelines
      with open(temp_file, 'r') as f:
        pipeline_args = json.load(f)
    else:
      # For non-TFX pipelines, extract pipeline name from the dsl filename.
      pipeline_args = {
          labels.PIPELINE_NAME:
              os.path.basename(self.flags_dict[labels.PIPELINE_DSL_PATH]
                              ).split('.')[0]
      }

    # Delete temp file
    io_utils.delete_dir(temp_file)

    return pipeline_args

  def _get_handler_home(self, home_dir: Text) -> Text:
    """Sets handler home.

    Args:
      home_dir: directory name to store pipelines

    Returns:
      Path to handler home directory.
    """
    engine_flag = self.flags_dict[labels.ENGINE_FLAG]
    handler_home_dir = engine_flag.upper() + '_HOME'
    if handler_home_dir in os.environ:
      return os.environ[handler_home_dir]
    return os.path.join(os.environ['HOME'], home_dir, '')

  def _subprocess_call(self,
                       command: List[Text],
                       env: Dict[Text, Any] = None) -> None:
    return_code = subprocess.call(command, env=env)
    if return_code != 0:
      sys.exit('Error while running "{}" '.format(' '.join(command)))
