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
"""Generate an example with variables.

******Too complicated to analyze at the beginning*******"""


import os
import numpy as np
import tensorflow as tf


def save_mobilenet_as_savedmodel(export_dir):
  file = tf.keras.utils.get_file(
      'grace_hopper.jp',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
  img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
  x = tf.keras.preprocessing.image.img_to_array(img)
  x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis])

  labels_path = tf.keras.utils.get_file(
      'ImageNetLabels.txt',
      'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
  imagenet_labels = np.array(open(labels_path).read().splitlines())

  pretrained_model = tf.keras.applications.MobileNet()
  result_before_save = pretrained_model(x)

  decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]
  print("Result before saving:\n", decoded)
  
  tf.saved_model.save(pretrained_model, export_dir)
  

import graph_partition
import tempfile


if __name__ == '__main__':
  with tempfile.TemporaryDirectory() as temp_dir:
    save_mobilenet_as_savedmodel(temp_dir)
        
    op_to_filepath = {'main': os.path.join(temp_dir, 'saved_model.pb')}
    op_to_graph_def, op_to_output_names = graph_partition.load_saved_models(
        op_to_filepath)
    op_to_execution_specs = graph_partition.partition_all_graphs(
        op_to_graph_def, op_to_output_names)
    
    print(len(op_to_execution_specs))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
