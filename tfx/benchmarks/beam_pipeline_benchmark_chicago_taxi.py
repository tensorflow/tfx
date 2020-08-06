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
"""TFMA, TFMA v2, TFT Apache Beam pipeline benchmarks for Chicago Taxi dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import beam_pipeline_benchmark_base
import os


class BeamPipelineBenchmarkChicagoTaxi(
    beam_pipeline_benchmark_base.BeamPipelineBenchmarkBase):
  """Runs Beam pipeline benchmarks on various runners"""

  def __init__(self, **kwargs):

    base_dir = os.environ['BASE_DIR']
    big_shuffle_input_file = os.environ['BIG_SHUFFLE_INPUT_FILE']
    big_shuffle_output_file = os.environ['BIG_SHUFFLE_OUTPUT_FILE']

    super(BeamPipelineBenchmarkChicagoTaxi, self).__init__(
        min_num_workers=1, max_num_workers=32,
        base_dir=base_dir,
        big_shuffle_input_file=big_shuffle_input_file,
        big_shuffle_output_file=big_shuffle_output_file,
        **kwargs)

if __name__ == "__main__":
  beam_pipeline_benchmark_chicago_taxi = BeamPipelineBenchmarkChicagoTaxi()

  beam_pipeline_benchmark_chicago_taxi.benchmarkLocalScaled()

  cloud_dataflow_project = os.environ['CLOUD_DATAFLOW_PROJECT']
  cloud_dataflow_temp_loc = os.environ['CLOUD_DATAFLOW_TEMP_LOC']
  beam_pipeline_benchmark_chicago_taxi.benchmarkCloudDataflow(
      cloud_dataflow_project, cloud_dataflow_temp_loc)

  beam_pipeline_benchmark_chicago_taxi.benchmarkFlinkOnK8s()
