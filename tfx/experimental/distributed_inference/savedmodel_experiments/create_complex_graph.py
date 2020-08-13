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
"""Save examples as SavedModels."""

import os
import tensorflow as tf

from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import create_complex_graph as create


def save_examples_as_saved_models(export_dir):
  """Saves the example graphs as SavedModels."""
  with tf.compat.v1.Session(graph=create.graph_a) as sess:
    tf.compat.v1.saved_model.simple_save(
        sess, os.path.join(export_dir, 'graph_a'),
        inputs={'input': create.ids_a},
        outputs={'output': create.result_a})

  with tf.compat.v1.Session(graph=create.graph_b) as sess:
    tf.compat.v1.saved_model.simple_save(
        sess, os.path.join(export_dir, 'graph_b'),
        inputs={'input': create.ids_b1, 'input_1': create.ids_b2},
        outputs={'output': create.result_b})

  with tf.compat.v1.Session(graph=create.main_graph) as sess:
    tf.compat.v1.saved_model.simple_save(
        sess, os.path.join(export_dir, 'main_graph'),
        inputs={'input': create.ids1, 'input_1': create.ids2},
        outputs={'output': create.main_result})

if __name__ == "__main__":
  save_examples_as_saved_models('./savedmodels')