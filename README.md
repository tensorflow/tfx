# TFX

TensorFlow Extended (TFX) is a TensorFlow-based general-purpose machine learning
platform implemented at Google. We’ve open sourced some of the TFX libraries
listed below. The [TFX repository](https://github.com/tensorflow/tfx) will
contain horizontal components that integrate these libraries in one platform.
For now, it will include examples of how to use the already open sourced TFX
libraries together.

## Examples

*   [Chicago Taxi Example](./examples/chicago_taxi)

## Available TFX libraries

### [TensorFlow Data Validation](https://github.com/tensorflow/data-validation)

A library for exploring and validating machine learning data.

### [TensorFlow Transform](https://github.com/tensorflow/transform)

A preprocessing pipeline to perform full-pass analyze phases over data to create
transformation graphs that are consistently applied during training and serving.

### [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis)

Libraries and visualization components to compute full-pass and sliced model
metrics over large datasets, and analyze them in a notebook.

### [TensorFlow Serving](https://github.com/tensorflow/serving)

A flexible, high-performance serving system for machine learning models,
designed for production environments.
