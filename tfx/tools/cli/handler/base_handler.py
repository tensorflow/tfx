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
import re
import sys

from six import with_metaclass
import tensorflow as tf
from tfx.tools.cli import labels


class BaseHandler(with_metaclass(abc.ABCMeta, object)):
  """Base Handler for CLI.

  Attributes:
    flags_dict: A dictionary with flags provided in a command.
  """
  # TODO(b/132286477): Update comments after finalizing return types.

  def __init__(self, flags_dict):
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
  def run_pipeline(self) -> None:
    """Runs a pipeline for the handler."""
    pass

  def _check_pipeline_dsl_path(self) -> None:
    """Check if pipeline dsl path exists."""
    # Check if dsl exists.
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
          'kubeflow': r'KubeflowDagRunner\(.*\)'
      }
      match = re.search(regexes[engine_flag], dsl_contents)
      if not match:
        sys.exit('{} runner not found in dsl.'.format(engine_flag))
