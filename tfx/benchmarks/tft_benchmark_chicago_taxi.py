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
"""TFT benchmark for Chicago Taxi dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from tfx.benchmarks import tft_benchmark_base
from tfx.benchmarks.datasets.chicago_taxi import dataset

from tensorflow.python.platform import test  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_analyzers_wide", 10,
                     "Number of analyzers in the TFT preprocessing function. "
                     "Only used in `TFTBenchmarkChicagoTaxiWide`.")


class TFTBenchmarkChicagoTaxi(tft_benchmark_base.TFTBenchmarkBase):

  def __init__(self, **kwargs):
    super(TFTBenchmarkChicagoTaxi, self).__init__(
        dataset=dataset.get_dataset(), **kwargs)


class TFTBenchmarkChicagoTaxiWide(tft_benchmark_base.TFTBenchmarkBase):

  def __init__(self, **kwargs):
    super(TFTBenchmarkChicagoTaxiWide, self).__init__(
        dataset=dataset.get_wide_dataset(num_analyzers=self._num_analyzers()),
        **kwargs)

  def _num_analyzers(self):
    return (FLAGS.num_analyzers_wide
            if FLAGS.is_parsed() else FLAGS["num_analyzers_wide"].default)


if __name__ == "__main__":
  test.main()
