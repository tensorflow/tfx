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
from tensorflow.python.platform import test  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "beam_runner", "DirectRunner",
    "Beam runner to use - any runner name accepted by "
    "apache_beam.runners.create_runner")
flags.DEFINE_integer(
    "max_num_examples", 0,
    "Maximum number of examples to read from the dataset. Use zero if there "
    "should be no limit")


class BenchmarkBase(test.Benchmark):
  """Base class for all benchmarks."""

  def _create_beam_pipeline(self):
    # FLAGS may not be parsed if the benchmark is instantiated directly by a
    # test framework (e.g. PerfZero creates the class and calls the methods
    # directly)
    runner_flag = (
        FLAGS.beam_runner
        if FLAGS.is_parsed() else FLAGS["beam_runner"].default)
    return beam.Pipeline(runner=beam.runners.create_runner(runner_flag))

  def _max_num_examples(self):
    result = (
        FLAGS.max_num_examples
        if FLAGS.is_parsed() else FLAGS["max_num_examples"].default)
    if result == 0:
      return None
    return result
