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
"""Abstract TFX executor class for Beam powered components."""

import sys
from typing import Any, Callable, List, Optional

from absl import flags
from absl import logging
from tfx.dsl.components.base.base_executor import BaseExecutor
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import telemetry_utils
from tfx.utils import dependency_utils

try:
  import apache_beam as beam  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


class BaseBeamExecutor(BaseExecutor):
  """Abstract TFX executor class for Beam powered components."""

  class Context(BaseExecutor.Context):
    """Context class for base Beam excecutor."""

    def __init__(
        self,
        beam_pipeline_args: Optional[List[str]] = None,
        extra_flags: Optional[List[str]] = None,
        tmp_dir: Optional[str] = None,
        unique_id: Optional[str] = None,
        executor_output_uri: Optional[str] = None,
        stateful_working_dir: Optional[str] = None,
        pipeline_node: Optional[pipeline_pb2.PipelineNode] = None,
        pipeline_info: Optional[pipeline_pb2.PipelineInfo] = None,
        pipeline_run_id: Optional[str] = None,
        make_beam_pipeline_fn: Optional[Callable[[], _BeamPipeline]] = None):
      super().__init__(
          extra_flags=extra_flags,
          tmp_dir=tmp_dir,
          unique_id=unique_id,
          executor_output_uri=executor_output_uri,
          stateful_working_dir=stateful_working_dir,
          pipeline_node=pipeline_node,
          pipeline_info=pipeline_info,
          pipeline_run_id=pipeline_run_id)
      self.beam_pipeline_args = beam_pipeline_args
      self.make_beam_pipeline_fn = make_beam_pipeline_fn

  def __init__(self, context: Optional[Context] = None):
    """Constructs a beam based executor."""
    super().__init__(context)

    self._beam_pipeline_args = None
    self._make_beam_pipeline_fn = None
    if context:
      if isinstance(context, BaseBeamExecutor.Context):
        self._beam_pipeline_args = context.beam_pipeline_args
        self._make_beam_pipeline_fn = context.make_beam_pipeline_fn
      else:
        raise ValueError('BaseBeamExecutor found initialized with '
                         'BaseExecutorSpec. Please use BeamExecutorSpec for '
                         'Beam Components instead.')

    if self._beam_pipeline_args:
      self._beam_pipeline_args = dependency_utils.make_beam_dependency_flags(
          self._beam_pipeline_args)
      executor_class_path = '%s.%s' % (self.__class__.__module__,
                                       self.__class__.__name__)
      # TODO(zhitaoli): Rethink how we can add labels and only normalize them
      # if the job is submitted against GCP.
      with telemetry_utils.scoped_labels(
          {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
        self._beam_pipeline_args.extend(
            telemetry_utils.make_beam_labels_args())

      # TODO(b/174174381): Don't use beam_pipeline_args to set ABSL flags.
      flags.FLAGS(sys.argv + self._beam_pipeline_args, known_only=True)

  # TODO(b/126182711): Look into how to support fusion of multiple executors
  # into same pipeline.
  def _make_beam_pipeline(self) -> _BeamPipeline:
    """Makes beam pipeline."""
    if self._make_beam_pipeline_fn is not None:
      return self._make_beam_pipeline_fn()
    if not beam:
      raise Exception(
          'Apache Beam must be installed to use this functionality.')

    result = beam.Pipeline(argv=self._beam_pipeline_args)

    # TODO(b/159468583): Obivate this code block by moving the warning to Beam.
    #
    # pylint: disable=g-import-not-at-top
    from apache_beam.options.pipeline_options import DirectOptions
    from apache_beam.options.pipeline_options import PipelineOptions
    options = PipelineOptions(self._beam_pipeline_args)
    direct_running_mode = options.view_as(DirectOptions).direct_running_mode
    direct_num_workers = options.view_as(DirectOptions).direct_num_workers
    if direct_running_mode == 'in_memory' and direct_num_workers != 1:
      logging.warning(
          'If direct_num_workers is not equal to 1, direct_running_mode should '
          'be `multi_processing` or `multi_threading` instead of `in_memory` '
          'in order for it to have the desired worker parallelism effect.')

    return result
