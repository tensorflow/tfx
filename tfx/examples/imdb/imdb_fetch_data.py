# Copyright 2021 Google LLC. All Rights Reserved.
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
"""A script to fetch IMDB data set."""

import os
import pandas as pd
import tensorflow_datasets as tfds

# Run this file to download and preprocess the entire imdb dataset.
# Remove the imdb_small_with_labels.csv that comes natively in the repo/data
# folder. Make sure imdb.csv is present in the /data folder.
# Change the hyperparameters to better suit the bigger dataset.
# The configurations that were found reasonable are listed below:
# imdb_pipeline_native_keras.py:
#    tfma.GenericValueThreshold(lower_bound={'value':0.85}
#    trainer_pb2.TrainArgs(num_steps=7000)
#    trainer_pb2.EvalArgs(num_steps=800)
# imdb_utils_native_keras.py:
#    _TRAIN_BATCH_SIZE=64
#    _EVAL_BATCH_SIZE=64

# Example use in another file of this directory:
# import imdb_fetch_data as full_data
# full_data.fetch_data()

# Dataset source acknowledgement:
# @InProceedings{maas-EtAl:2011:ACL-HLT2011,
#   author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T. and
#   Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
#   title     = {Learning Word Vectors for Sentiment Analysis},
#   booktitle = {Proceedings of the 49th Annual Meeting of the Association for
#   Computational Linguistics: Human Language Technologies},
#   month     = {June},
#   year      = {2011},
#   address   = {Portland, Oregon, USA},
#   publisher = {Association for Computational Linguistics},
#   pages     = {142--150},
#   url       = {http://www.aclweb.org/anthology/P11-1015}
# }


def fetch_data():
  """This downloads the full dataset to $(pwd)/data/imdb.csv."""

  ds = tfds.load('imdb_reviews', split='train+test')
  numpy_ds = tfds.as_numpy(ds)
  df = pd.DataFrame(numpy_ds)
  df['text'] = df['text'].str.decode('utf-8')
  dst_path = os.getcwd() + '/data/imdb.csv'
  df.to_csv(dst_path, index=False)


if __name__ == '__main__':
  fetch_data()
