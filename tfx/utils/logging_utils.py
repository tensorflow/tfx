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

import logging
import os
import tensorflow as tf
from typing import Text


def get_logger(log_dir,
               log_file_name,
               logging_level = logging.INFO):
  """Create and configure a TFX-specific logger.

  Args:
    log_dir: log directory path, will create if not exists.
    log_file_name: name of log file under above directory.
    logging_level: logger's level, default to INFO.

  Returns:
    A logger that outputs to log_dir/log_file_name.

  Raises:
    RuntimeError: if log dir exists as a file.

  """
  log_path = os.path.join(log_dir, log_file_name)
  logger = logging.getLogger(log_path)
  logger.setLevel(logging_level)

  if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)
  if not tf.gfile.IsDirectory(log_dir):
    raise RuntimeError('Log dir exists as a file: {}'.format(log_dir))

  # Create logfile handler.
  fh = logging.FileHandler(log_path)
  # Define logmsg format.
  formatter = logging.Formatter(
      '%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s: %(message)s')
  fh.setFormatter(formatter)
  # Add handler to logger.
  logger.addHandler(fh)

  return logger
