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
"""Tool to regenerate datasets used in benchmarks."""

# Standard Imports

from absl import app
from absl import flags
from absl import logging

from tfx.benchmarks import benchmark_utils
from tfx.benchmarks import tft_benchmark_base

FLAGS = flags.FLAGS


def main(argv):
  del argv
  dataset = benchmark_utils.get_dataset(FLAGS.dataset,
                                        base_dir=FLAGS.output_base_dir)

  # Regenerate the dataset and models.
  logging.info("Using dataset: %s", FLAGS.dataset)
  logging.info("Generating raw dataset")
  dataset.generate_raw_dataset(args=FLAGS.generate_dataset_args)
  logging.info("Generating models")
  logging.info("force_tf_compat_v1: %s", FLAGS.force_tf_compat_v1)
  dataset.generate_models(
      args=FLAGS.generate_dataset_args,
      force_tf_compat_v1=FLAGS.force_tf_compat_v1)

  # Regenerate intermediate outputs for TFT benchmarks.
  logging.info("Generating intermediate outputs for TFT benchmarks")
  tft_benchmark_base.regenerate_intermediates_for_dataset(
      dataset, force_tf_compat_v1=FLAGS.force_tf_compat_v1)


if __name__ == "__main__":
  flags.DEFINE_string("dataset", "chicago_taxi", "Dataset to run on.")
  flags.DEFINE_string("output_base_dir", "", "Base directory under which to "
                      "write generated Dataset artifacts.")
  flags.DEFINE_string(
      "generate_dataset_args", "",
      "Arguments to pass to the dataset when regenerating the raw dataset or "
      "the models.")
  flags.DEFINE_bool(
      "force_tf_compat_v1", True,
      "If False, TFT will use native TF 2 implementation if TF 2 behaviors are "
      "enabled.")
  app.run(main)
