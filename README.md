<!-- See: www.tensorflow.org/tfx/ -->

# TFX

[![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)](https://github.com/tensorflow/tfx)
[![PyPI](https://badge.fury.io/py/tfx.svg)](https://badge.fury.io/py/tfx)

TensorFlow Extended (TFX) is a TensorFlow-based general-purpose machine learning
platform implemented at Google. We’ve open sourced some of the TFX libraries
listed below. The [TFX repository](https://github.com/tensorflow/tfx) will
contain horizontal components that integrate these libraries in one platform.
For now, it will include examples of how to use the already open sourced TFX
libraries together.

## Examples

*   [Chicago Taxi Example](https://github.com/tensorflow/tfx/tree/master/examples/chicago_taxi_pipeline)

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

## Compatible versions

The following table describes how the `tfx` package versions are compatible
with its major dependency Pypi packages. This is determined by our testing
framework, but other *untested* combinations may also work.

|tfx                                                                                |tensorflow    |apache-beam[gcp]|tensorflow-data-validation|tensorflow-model-analysis|tensorflow-transform|ml-metadata|tensorflow-metadata|
|-----------------------------------------------------------------------------------|--------------|----------------|--------------------------|--------------------------------|--------------------|-----------|-------------------|
|[GitHub master](https://github.com/tensorflow/tfx/blob/master/tfx/g3doc/RELEASE.md)|nightly (1.x) |2.10.0          |0.12.0                    |0.12.1                           |0.12.0              |0.13.2     |0.12.1             |
