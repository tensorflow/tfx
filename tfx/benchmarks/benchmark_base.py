# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Base class for benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import apache_beam as beam
from apache_beam.pipeline import PipelineOptions
from tensorflow.python.platform import test  # pylint: disable=g-direct-tensorflow-import
from tfx.benchmarks import mode_config

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "beam_runner", "DirectRunner",
    "Beam runner to use - any runner name accepted by "
    "apache_beam.runners.create_runner")


class BenchmarkBase(test.Benchmark):
  """Base class for running Beam pipelines on various runners"""

  def __init__(self):
    super(BenchmarkBase, self).__init__()
    self.beam_pipeline_mode = mode_config.DEFAULT_MODE
    self.num_workers = 1
    self.cloud_dataflow_temp_loc = None
    self.cloud_dataflow_project = None

  def _set_cloud_dataflow_options(self):
    self.pipeline_options = PipelineOptions(runner="DataflowRunner",
                                            project=self.cloud_dataflow_project,
                                            temp_location=self.cloud_dataflow_temp_loc,
                                            num_workers=self.num_workers,
                                            no_pipeline_type_check=True,
                                            setup_file="./setup.py",
                                            autoscaling_algorithm="NONE",
                                            region="us-central1")

  def _set_flink_on_k8s_operator_options(self):
    self.pipeline_options = PipelineOptions(runner="PortableRunner",
                                            job_endpoint="localhost:8099",
                                            artifact_endpoint="localhost:8098",
                                            environment_type="EXTERNAL",
                                            environment_config="localhost:50000",
                                            parallelism=self.num_workers,
                                            no_pipeline_type_check=True)

  def _set_local_scaled_execution_options(self):
    self.pipeline_options = PipelineOptions(runner="DirectRunner",
                                            direct_running_mode="multi_processing",
                                            direct_num_workers=self.num_workers,
                                            no_pipeline_type_check=True)

  def set_beam_pipeline_mode(self, beam_pipeline_mode):
    assert beam_pipeline_mode in mode_config.modes
    self.beam_pipeline_mode = beam_pipeline_mode

  def set_num_workers(self, num_workers):
    self.num_workers = num_workers

  def set_cloud_dataflow_temp_loc(self, cloud_dataflow_temp_loc):
    self.cloud_dataflow_temp_loc = cloud_dataflow_temp_loc

  def set_cloud_dataflow_project(self, cloud_dataflow_project):
    self.cloud_dataflow_project = cloud_dataflow_project

  def _create_beam_pipeline_default(self):
    # FLAGS may not be parsed if the benchmark is instantiated directly by a
    # test framework (e.g. PerfZero creates the class and calls the methods
    # directly)
    runner_flag = (
        FLAGS.beam_runner
        if FLAGS.is_parsed() else FLAGS["beam_runner"].default)
    return beam.Pipeline(runner=beam.runners.create_runner(runner_flag))

  def _create_beam_pipeline(self):
    if self.beam_pipeline_mode == mode_config.LOCAL_SCALED_EXECUTION_MODE:
      self._set_local_scaled_execution_options()

    elif self.beam_pipeline_mode == mode_config.CLOUD_DATAFLOW_MODE:
      self._set_cloud_dataflow_options()

    elif self.beam_pipeline_mode == mode_config.FLINK_ON_K8S_MODE:
      self._set_flink_on_k8s_operator_options()

    else:
      return self._create_beam_pipeline_options()

    return beam.Pipeline(options=self.pipeline_options)
