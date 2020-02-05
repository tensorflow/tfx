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
"""Utils for TFX-specific logger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import os
from typing import Any, Dict, Optional, Text

import tensorflow.compat.v1 as tf


class LoggerConfig(object):
  """Logger configuration class.

  Logger configuration consists of:
    - pipeline_name: name of active pipeline
    - worker_name: name of component/object doing the logging
    - log_root: path for log directory
    - log_level: logger's level, default to INFO.
  """

  def __init__(self,
               log_root: Optional[Text] = '/var/tmp/tfx/logs',
               log_level: Optional[int] = logging.INFO,
               pipeline_name: Optional[Text] = '',
               worker_name: Optional[Text] = ''):
    self.log_root = log_root
    self.log_level = log_level
    self.pipeline_name = pipeline_name
    self.worker_name = worker_name

  def update(self, config: Optional[Dict[Text, Any]] = None):
    """Updates the log config parameters via elements in a dict.

    Args:
      config: Dict of parameter tuples to assign to the logging config.
    Raises:
      ValueError if key is not a supported logging parameter.
    """
    if config:
      for k, v in config.items():
        if k in ('log_root', 'log_level', 'pipeline_name', 'worker_name'):
          setattr(self, k, v)
        else:
          raise ValueError('%s not expected in logger config.' % k)

  def copy(self):
    """Returns a shallow copy of this config."""
    return copy.copy(self)


def get_logger(config):
  """Create and configure a TFX-specific logger.

  Args:
    config: LoggingConfig class used to configure logger
  Returns:
    A logger that outputs to log_dir/log_file_name.
  Raises:
    RuntimeError: if log dir exists as a file.

  """
  log_path = os.path.join(config.log_root, 'tfx.log')
  logger = logging.getLogger(log_path)
  logger.setLevel(config.log_level)

  if not tf.io.gfile.exists(config.log_root):
    tf.io.gfile.makedirs(config.log_root)
  if not tf.io.gfile.isdir(config.log_root):
    raise RuntimeError('Log dir exists as a file: {}'.format(config.log_root))

  # Create logfile handler.
  fh = logging.FileHandler(log_path)
  # Define logmsg format.
  formatter = logging.Formatter(
      '%(asctime)s - {}:{} (%(filename)s:%(lineno)s) - %(levelname)s: %(message)s'
      .format(config.pipeline_name, config.worker_name))
  fh.setFormatter(formatter)
  # Add handler to logger.
  logger.addHandler(fh)

  return logger
