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

from tfx.benchmarks import beam_pipeline_benchmark_base


class BeamPipelineBenchmarkChicagoTaxi(
    beam_pipeline_benchmark_base.BeamPipelineBenchmarkBase):

  def __init__(self, **kwargs):
    super(BeamPipelineBenchmarkChicagoTaxi, self).__init__(
        min_num_workers=3, max_num_workers=4,
        base_dir="gs://tfx-keshav-example-bucket/datasets",
        cloud_dataflow_temp_loc="gs://tfx-keshav-example-bucket/temp",
        **kwargs)


if __name__ == "__main__":
  beam_pipeline_benchmark_chicago_taxi = BeamPipelineBenchmarkChicagoTaxi()
  beam_pipeline_benchmark_chicago_taxi.benchmarkLocalScaled()
  beam_pipeline_benchmark_chicago_taxi.benchmarkCloudDataflow()
  beam_pipeline_benchmark_chicago_taxi.benchmarkFlinkOnK8s()
