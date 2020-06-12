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
"""Executor specifications for defining what to to execute."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import List, Text, Type

from six import with_metaclass

from tfx.components.base import base_executor
from tfx.utils import import_utils
from tfx.utils import json_utils


class ExecutorSpec(with_metaclass(abc.ABCMeta, json_utils.Jsonable)):
  """A specification for a component executor.

  An instance of ExecutorSpec describes the implementation of a component.
  """


class ExecutorClassSpec(ExecutorSpec):
  """A specification of executor class.

  Attributes:
    executor_class: a subclass of base_executor.BaseExecutor used to execute
      this component (required).
  """

  def __init__(self, executor_class: Type[base_executor.BaseExecutor]):
    if not executor_class:
      raise ValueError('executor_class is required')
    self.executor_class = executor_class
    super(ExecutorClassSpec, self).__init__()

  def __reduce__(self):
    # When executing on the Beam DAG runner, the ExecutorClassSpec instance
    # is pickled using the "dill" library. To make sure that the executor code
    # itself is not pickled, we save the class path which will be reimported
    # by the worker in this custom __reduce__ function.
    #
    # See https://docs.python.org/3/library/pickle.html#object.__reduce__ for
    # more details.
    executor_class_path = '%s.%s' % (self.executor_class.__module__,
                                     self.executor_class.__name__)
    return (ExecutorClassSpec._reconstruct_from_executor_class_path,
            (executor_class_path,))

  @staticmethod
  def _reconstruct_from_executor_class_path(executor_class_path):
    executor_class = import_utils.import_class_by_path(executor_class_path)
    return ExecutorClassSpec(executor_class)


class ExecutorContainerSpec(ExecutorSpec):
  """A specification of a container.

  The spec includes image, command line entrypoint and arguments for a
  container. For example:

  spec = ExecutorContainerSpec(
    image='docker/whalesay',
    command=['cowsay'],
    args=['hello wolrd'])

  Attributes:
    image: Container image that has executor application. Assumption is that
      this container image is separately release-managed, and tagged/versioned
      accordingly.
    command: Container entrypoint array. Not executed within a shell. The docker
      image's ENTRYPOINT is used if this is not provided. The Jinja templating
      mechanism is used for constructing a user-specified command-line
      invocation based on input and output metadata at runtime.
    args: Arguments to the container entrypoint. The docker image's CMD is used
      if this is not provided. The Jinja templating mechanism is used for
      constructing a user-specified command-line invocation based on input and
      output metadata at runtime.
  """

  def __init__(self,
               image: Text,
               command: List[Text] = None,
               args: List[Text] = None):
    if not image:
      raise ValueError('image cannot be None or empty.')
    self.image = image
    self.command = command
    self.args = args
    super(ExecutorContainerSpec, self).__init__()
