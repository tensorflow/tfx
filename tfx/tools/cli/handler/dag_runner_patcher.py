# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Base class to patch DagRunner classes in TFX CLI."""

import abc
import contextlib
import functools
from typing import Any, MutableMapping, Type, Union

from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.portable import tfx_runner as portable_tfx_runner
from tfx.proto.orchestration import pipeline_pb2


class DagRunnerPatcher(abc.ABC):
  """Abstract base class for Patchers for various "DagRunner"s.

  These patcher classes "decorate" the `run` function of the DagRunners.
  The patcher can intercept arguments before/after the run() execution,
  and even skip the real run() execution.
  Users of the patcher can read information about the run using `context`
  which is passed as a context manager of patch(). For example,
  the name of the pipeline can be retrieved with
  `context[DagRunnerPatcher.PIPELINE_NAME]`.

  Child class should override get_runner_class() to specify which DagRunner
  class they patch. And they can add more actions in before_run()/after_run()
  as they needed.
  """

  PIPELINE_NAME = 'pipeline_name'
  PIPELINE_ROOT = 'pipeline_root'

  def __init__(self, call_real_run=True):
    """Init method of base class.

    Do NOT use this directly.
    Please use subclass of DagRunnerPatcher.

    Args:
      call_real_run: Specify DagRunner.run() should be called or bypassed.
    """
    self._context = {}
    self._run_called = False
    self._call_real_run = call_real_run

  def _before_run(self, runner: tfx_runner.TfxRunner,
                  pipeline: Union[pipeline_pb2.Pipeline, tfx_pipeline.Pipeline],
                  context: MutableMapping[str, Any]) -> None:
    pass

  def _after_run(self, runner: tfx_runner.TfxRunner,
                 pipeline: Union[pipeline_pb2.Pipeline, tfx_pipeline.Pipeline],
                 context: MutableMapping[str, Any]) -> None:
    pass

  @abc.abstractmethod
  def get_runner_class(
      self
  ) -> Union[Type[tfx_runner.TfxRunner], Type[portable_tfx_runner.TfxRunner]]:
    raise NotImplementedError()

  @property
  def run_called(self) -> bool:
    return self._run_called

  @contextlib.contextmanager
  def patch(self):
    """Context manager to patch the run function.

    Example usage:
      patcher = SomeDagRunnerPatcher()
      with patcher.patch() as context:
        runner.run(pipeline)
        print( context[patcher.PIPELINE_NAME] ) ...

    Yields:
      a MutableMapping that holds information on the pipeline.
    """

    runner_class = self.get_runner_class()
    old_run = runner_class.run
    runner_class.run = self._decorate(runner_class.run)
    try:
      yield self._context
    finally:
      runner_class.run = old_run

  def _decorate(self, f):
    """Decorate the run function."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      # Make sure that the run() is only called exactly once.
      if self._run_called:
        raise RuntimeError(
            f'{self.get_runner_class().__name__}.run() called more than once.')
      self._run_called = True

      assert len(args) >= 1, 'run() must have self.'
      runner = args[0]
      if 'pipeline' in kwargs:
        pipeline = kwargs['pipeline']
      else:
        assert len(args) >= 2, 'run() must have pipeline argument.'
        pipeline = args[1]

      if isinstance(pipeline, tfx_pipeline.Pipeline):
        self._context[self.PIPELINE_NAME] = pipeline.pipeline_info.pipeline_name
        self._context[self.PIPELINE_ROOT] = pipeline.pipeline_info.pipeline_root
      else:  # pipeline_pb2.Pipeline
        self._context[self.PIPELINE_NAME] = pipeline.pipeline_info.id
        self._context[self.PIPELINE_ROOT] = (
            pipeline.runtime_spec.pipeline_root.runtime_parameter.default_value
            .string_value)

      self._before_run(runner, pipeline, self._context)
      if self._call_real_run:
        result = f(*args, **kwargs)
      else:
        result = None
      self._after_run(runner, pipeline, self._context)
      return result

    return wrapper
