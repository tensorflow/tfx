# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Supplement for flowers_utils_base.py with specifics for Keras models.

This module file will be used in the Transform, Tuner and generic Trainer
components.
"""

import os
from abc import ABC
from typing import Dict, Union, List, Text

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras import layers, Input, Sequential, Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss
from tensorflow import Tensor
import tensorflow.keras as keras

from tfx.examples.flowers_GAN import flowers_utils_base as base
from tfx.examples.flowers_GAN.flowers_utils_base import transformed_name, _LABEL_KEY

# TFX Transform will call this function.
preprocessing_fn = base.preprocessing_fn

INPUT_SHAPE_GENERATOR = 300
BATCH_SIZE = 64
MODEL_VERSION = 1


def _gzip_reader_fn(
        filenames: Union[type(tf.string), type(tf.data.Dataset)]) -> tf.data.TFRecordDataset:
    """Function to read in GZipped TFRecords as datasets."""
    return tf.data.TFRecordDataset(filenames,
                                   compression_type='GZIP')


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = BATCH_SIZE):
    """Generates features and label for training.
    Args:
      file_pattern: List of paths or patterns of input tfrecord files.
      tf_transform_output: A TFTransformOutput.
      batch_size: representing the number of consecutive elements of returned
        dataset to combine in a single batch
    Returns:
      A dataset that contains (features, indices) tuple where features is a
        dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
    )

    return dataset


def _get_serve_tf_examples_fn(model, tf_transform_output: tft.TFTransformOutput):
    """Returns a function that parses a serialized tf.Example.
    Used for serving a model."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)

        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_serve_tf_examples_fn_generator(model):
    """Returns a function that parses a serialized tf.Example.
    Serving function for a GAN generator that receives noise as input."""

    @tf.function
    def serve_tf_examples_fn(noise):
        """Returns the output to be used in the serving signature."""
        outputs = model(noise)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def make_generator_model() -> Model:
    """Function to create a Keras GAN generator model."""
    model = Sequential()
    model.add(layers.Dense(14 * 14 * 256, use_bias=False, input_shape=(INPUT_SHAPE_GENERATOR,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((14, 14, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                     activation='tanh'))
    assert model.output_shape == (None, 56, 56, 3)

    return model


def make_discriminator_model() -> Model:
    """Function to create a Keras GAN discriminator model."""
    input_layer = Input(shape=[56, 56, 3], name=transformed_name(_LABEL_KEY))

    layer = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_layer)
    layer = layers.LeakyReLU()(layer)
    layer = layers.Dropout(0.3)(layer)

    layer = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(layer)
    layer = layers.LeakyReLU()(layer)
    layer = layers.Dropout(0.3)(layer)

    layer = layers.Flatten()(layer)
    output = layers.Dense(1)(layer)

    model = Model(input_layer, output)

    return model


class GAN(keras.Model, ABC):
    """GAN Keras model. More info:
    https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#wrapping_up_an_end-to
    -end_gan_example """

    def __init__(self,
                 discriminator: Model,
                 generator: Model,
                 latent_dim: int):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self,
                d_optimizer: Optimizer,
                g_optimizer: Optimizer,
                loss_fn: Loss):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images: Tensor) -> Dict[str, Tensor]:
        random_latent_vectors = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        real_images = real_images[transformed_name(_LABEL_KEY)]
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((BATCH_SIZE, 1)), tf.zeros((BATCH_SIZE, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(BATCH_SIZE, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((BATCH_SIZE, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


def get_model() -> GAN:
    """Function to generate GAN object, compile with optimizers/loss and return it."""
    gan = GAN(discriminator=make_discriminator_model(), generator=make_generator_model(),
              latent_dim=INPUT_SHAPE_GENERATOR)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))

    return gan


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)

    gan = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    gan.fit(train_dataset,
            steps_per_epoch=fn_args.train_steps,
            callbacks=[tensorboard_callback])

    signatures_generator = {
        'serving_default':
            _get_serve_tf_examples_fn_generator(gan.generator).get_concrete_function(tf.TensorSpec(
                shape=[None, INPUT_SHAPE_GENERATOR],
                dtype=tf.float32,
                name='examples')),
    }

    signatures_discriminator = {
        'serving_default':
            _get_serve_tf_examples_fn(gan.discriminator,
                                      tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')),
    }
    gan.generator.save(
        os.path.join(os.path.join(fn_args.serving_model_dir, 'generator'), str(MODEL_VERSION)),
        save_format='tf', signatures=signatures_generator)
    gan.discriminator.save(
        os.path.join(os.path.join(fn_args.serving_model_dir, 'discriminator'), str(MODEL_VERSION)),
        save_format='tf', signatures=signatures_discriminator)
