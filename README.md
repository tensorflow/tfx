<!-- See: www.tensorflow.org/tfx/ -->

# TFX

[![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)](https://github.com/tensorflow/tfx)
[![PyPI](https://badge.fury.io/py/tfx.svg)](https://badge.fury.io/py/tfx)

[TensorFlow Extended (TFX)](https://tensorflow.org/tfx) is a
Google-production-scale machine learning platform based on TensorFlow. It
provides a configuration framework to express ML pipelines consisting of TFX
components. TFX pipelines can be orchestrated using
[Apache Airflow](https://airflow.apache.org/) and
[Kubeflow Pipelines](https://www.kubeflow.org/). Both the components themselves
as well as the integrations with orchestration systems can be extended.

TFX components interact with a
[ML Metadata](https://github.com/google/ml-metadata) backend that keeps a record
of component runs, input and output artifacts, and runtime configuration. This
metadata backend enables advanced functionality like experiment tracking or
warmstarting/resuming ML models from previous runs.

![TFX Components](https://raw.github.com/tensorflow/tfx/master/docs/guide/diag_all.svg?sanitize=true)

## Documentation

Please see the
[TFX User Guide](https://github.com/tensorflow/tfx/blob/master/docs/guide/index.md).

## Examples

*   [Chicago Taxi Example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline)

## Compatible versions

The following table describes how the `tfx` package versions are compatible with
its major dependency PyPI packages. This is determined by our testing framework,
but other *untested* combinations may also work.

tfx                                                                                 | tensorflow    | tensorflow-data-validation | tensorflow-model-analysis | tensorflow-metadata | tensorflow-transform | ml-metadata | apache-beam[gcp]
----------------------------------------------------------------------------------- | ------------- | -------------------------- | ------------------------- | ------------------- | -------------------- | ----------- | ----------------
[GitHub master](https://github.com/tensorflow/tfx/blob/master/tfx/g3doc/RELEASE.md) | nightly (1.x) | 0.13.1                     | 0.13.2                    | 0.13.0              | 0.13.0               | 0.13.2      | 2.12.0
0.13.0                                                                              | 1.13.1        | 0.13.1                     | 0.13.2                    | 0.13.0              | 0.13.0               | 0.13.2      | 2.12.0
0.12.0                                                                              | 1.12          | 0.12.0                     | 0.12.1                    | 0.12.1              | 0.12.0               | 0.13.2      | 2.10.0
