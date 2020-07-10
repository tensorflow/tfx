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
"""TFMA, TFMA v2, TFT Apache Beam pipeline benchmarks for Chicago Taxi dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("/home/kshivvy/tfx-benchmarks-PR/tfx/")

from tensorflow.python.platform import test  # pylint: disable=g-direct-tensorflow-import
import beam_pipeline_benchmark_base

class BeamPipelineBenchmarkBaseChicagoTaxi(beam_pipeline_benchmark_base.BeamPipelineBenchmarkBase):

  def __init__(self, **kwargs):
    super(BeamPipelineBenchmarkBaseChicagoTaxi, self).__init__(
        min_num_workers=1, max_num_workers=32,
        base_dir="gs://tfx-keshav-example-bucket/datasets",
        temp_location_for_cloud_dataflow="gs://tfx-keshav-example-bucket/temp",
        **kwargs)


if __name__ == "__main__":
  test.main()
