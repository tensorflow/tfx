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
"""TFMA v2 benchmark for Chicago Taxi dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test  # pylint: disable=g-direct-tensorflow-import
from tfx.benchmarks import tfma_v2_benchmark_base
from tfx.benchmarks.datasets.chicago_taxi import dataset


class TFMAV2BenchmarkChicagoTaxi(tfma_v2_benchmark_base.TFMAV2BenchmarkBase):

  def __init__(self, **kwargs):
    super(TFMAV2BenchmarkChicagoTaxi, self).__init__(
        dataset=dataset.get_dataset(), **kwargs)


if __name__ == "__main__":
  test.main()
