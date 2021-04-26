# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source which includes pipeline functions for the flowers dataset.

The utilities in this file are used to build a model with native Keras or with
Flax.
"""

from typing import Dict, Any
import tensorflow as tf

_LABEL_KEY = 'image_raw'

IMAGE_TARGET_HEIGHT = 56
IMAGE_TARGET_WIDTH = 56


def transformed_name(key: str) -> str:
    """Function to return processed keys."""
    return key + '_xf'


def process_rgb_image(raw_image: bytes) -> tf.Tensor:
    """Function to process byte strings of images to resized image tensor."""
    raw_image = tf.reshape(raw_image, [])
    img_rgb = tf.image.decode_jpeg(raw_image, channels=3)
    img = tf.image.convert_image_dtype(img_rgb, tf.float32)
    resized_img = tf.image.resize_with_pad(
        img,
        target_height=IMAGE_TARGET_HEIGHT,
        target_width=IMAGE_TARGET_WIDTH,
    )
    return resized_img


# TFX Transform will call this function.
def preprocessing_fn(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocessing function that follows the TFX requirements.
    For more information how this function could be set up: notebooks/read-tfrecords-image.ipynb.

    Args:
        inputs: dictionary containing TFRecords data metrics.
    """
    image_key = 'image_raw'

    image_raw = inputs[image_key]
    img_preprocessed = tf.map_fn(process_rgb_image, image_raw, dtype=tf.float32)

    return {
        transformed_name(image_key): img_preprocessed,
    }
