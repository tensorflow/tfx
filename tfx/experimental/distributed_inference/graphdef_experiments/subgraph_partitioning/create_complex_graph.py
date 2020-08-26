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
"""Generate an example consists of {'main', 'remote_op_a', 'remote_op_b'}.

When calling a remote op, we feed in some inputs, execute the graph associated
with the remote op, and get an output.

In this example, main calls each of remote_op_a and remote_op_b two times,
remote_op_b calls remote_op_a two times, and remote_op_a calls no one.
"""

import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # Disable eager mode

N = 1000  # number of embeddings
NDIMS = 16  # dimensionality of embeddings


def create_session(graph):
  return tf.compat.v1.Session(
      graph=graph,
      config=tf.compat.v1.ConfigProto(inter_op_parallelism_threads=8))


# Define remote_op_a's graph
graph_a = tf.Graph()
with graph_a.as_default():
  table_a = tf.random.uniform(shape=[N, NDIMS], seed=10)
  ids_a = tf.compat.v1.placeholder(dtype=tf.int32, name='ids_a')
  result_a = tf.nn.embedding_lookup(table_a, ids_a)


def remote_op_a(input_ids):
  """Mimics a remote op by numpy_function."""

  def remote_lookup(input_ids):
    with create_session(graph_a) as sess:
      return sess.run(result_a, feed_dict={ids_a: input_ids})

  return tf.compat.v1.numpy_function(
      func=remote_lookup, inp=[input_ids], Tout=tf.float32, name='remote_op_a')


# Define remote_op_b's graph
graph_b = tf.Graph()
with graph_b.as_default():
  ids_b2 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids_b2')
  ids_b1 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids_b1')
  ids_b1_preprocessed = tf.math.floormod(tf.add(ids_b1, 1), N)

  remote_result_a1 = remote_op_a(ids_b1_preprocessed)
  remote_result_a2 = remote_op_a(ids_b2)
  result_b = tf.math.add(remote_result_a1, remote_result_a2 * 2.5)


def remote_op_b(input_ids1, input_ids2):
  """Mimics another remote op."""

  def remote_lookup(input_ids1, input_ids2):
    with create_session(graph_b) as sess:
      return sess.run(
          result_b, feed_dict={
              ids_b1: input_ids1,
              ids_b2: input_ids2
          })

  return tf.compat.v1.numpy_function(
      func=remote_lookup,
      inp=[input_ids1, input_ids2],
      Tout=tf.float32,
      name='remote_op_b')


# Define main's graph
main_graph = tf.Graph()
with main_graph.as_default():
  ids1 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids1')
  ids2 = tf.compat.v1.placeholder(dtype=tf.int32, name='ids2')
  casted_ids1 = tf.cast(ids1, tf.float32)
  casted_ids2 = tf.cast(ids2, tf.float32)

  remote_a0 = remote_op_a(ids1)
  remote_b0 = remote_op_b(ids1, ids2)

  left_upper_concat = tf.concat([remote_a0, remote_b0], 0)
  left_upper_sum = tf.reduce_mean(left_upper_concat)

  right_upper_sum = tf.reduce_mean(remote_b0)
  right_upper_mul = tf.multiply(right_upper_sum, casted_ids2)
  right_upper_add = tf.add(right_upper_mul, left_upper_sum)
  right_upper_round = tf.math.round(right_upper_mul)
  right_upper_floormod = tf.math.floormod(right_upper_round, N)

  left_upper_add = tf.add_n([left_upper_sum, casted_ids1, right_upper_add])
  left_upper_round = tf.math.round(left_upper_add)
  left_upper_floormod = tf.math.floormod(left_upper_round, N)

  remote_a1 = remote_op_a(left_upper_floormod)
  remote_b1 = remote_op_b(left_upper_floormod, right_upper_floormod)

  left_lower_sum = tf.reduce_mean(remote_a1)

  right_lower_sum = tf.reduce_mean(remote_b1)
  right_lower_mul = tf.multiply(casted_ids2, right_lower_sum)
  right_lower_div = tf.divide(right_upper_add, right_lower_mul)

  main_result = tf.add_n([
      left_lower_sum, right_lower_div, right_lower_sum, right_upper_sum,
      tf.cast(left_upper_floormod, tf.float32)
  ])


def save_examples_as_graphdefs(export_dir):
  tf.io.write_graph(
      graph_a.as_graph_def(), export_dir, 'graph_a.pb', as_text=False)
  tf.io.write_graph(
      graph_b.as_graph_def(), export_dir, 'graph_b.pb', as_text=False)
  tf.io.write_graph(
      main_graph.as_graph_def(), export_dir, 'main_graph.pb', as_text=False)


if __name__ == '__main__':
  save_examples_as_graphdefs('./complex_graphdefs')
