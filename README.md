<!-- See: www.tensorflow.org/tfx/ -->

# TFX

[![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)](https://github.com/tensorflow/tfx)
[![PyPI](https://badge.fury.io/py/tfx.svg)](https://badge.fury.io/py/tfx)
[![TensorFlow](https://img.shields.io/badge/TensorFow-page-orange)](https://www.tensorflow.org/tfx)

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

![TFX Components](https://raw.githubusercontent.com/tensorflow/tfx/master/docs/guide/images/prog_fin.png)

## Documentation

### User Documentation

Please see the
[TFX User Guide](https://github.com/tensorflow/tfx/blob/master/docs/guide/index.md).

### Development References

#### Roadmap

The TFX [Roadmap](https://github.com/tensorflow/tfx/blob/master/ROADMAP.md),
which is updated quarterly.

#### Release Details

For detailed previous and upcoming changes, please
[check here](https://github.com/tensorflow/tfx/blob/master/RELEASE.md)

#### Requests For Comment

TFX is an open-source project and we strongly encourage active participation
by the ML community in helping to shape TFX to meet or exceed their needs. An
important component of that effort is the RFC process.  Please see the listing
of [current and past TFX RFCs](RFCs.md). Please see the
[TensorFlow Request for Comments (TF-RFC)](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md)
process page for information on how community members can contribute.

## Examples

*   [Chicago Taxi Example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline)

## Compatible versions

The following table describes how the `tfx` package versions are compatible with
its major dependency PyPI packages. This is determined by our testing framework,
but other *untested* combinations may also work.

tfx                                                                       | apache-beam[gcp] | ml-metadata | pyarrow | tensorflow        | tensorflow-data-validation | tensorflow-metadata | tensorflow-model-analysis | tensorflow-serving-api | tensorflow-transform | tfx-bsl
------------------------------------------------------------------------- | ---------------- | ----------- | ------- | ----------------- | -------------------------- | ------------------- | ------------------------- | ---------------------- | -------------------- | -------
[GitHub master](https://github.com/tensorflow/tfx/blob/master/RELEASE.md) | 2.32.0           | 1.3.0       | 2.0.0   | nightly (1.x/2.x) | 1.3.0                      | 1.2.0               | 0.34.1                    | 2.6.0                  | 1.3.0                | 1.3.0
1.3.1                                                                     | 2.32.0           | 1.3.0       | 2.0.0   | 1.15.0 / 2.6.0    | 1.3.0                      | 1.2.0               | 0.34.1                    | 2.6.0                  | 1.3.0                | 1.3.0
1.3.0                                                                     | 2.32.0           | 1.3.0       | 2.0.0   | 1.15.0 / 2.6.0    | 1.3.0                      | 1.2.0               | 0.34.1                    | 2.6.0                  | 1.3.0                | 1.3.0
1.2.0                                                                     | 2.31.0           | 1.2.0       | 2.0.0   | 1.15.0 / 2.5.0    | 1.2.0                      | 1.2.0               | 0.33.0                    | 2.5.1                  | 1.2.0                | 1.2.0
1.0.0                                                                     | 2.29.0           | 1.0.0       | 2.0.0   | 1.15.0 / 2.5.0    | 1.0.0                      | 1.0.0               | 0.31.0                    | 2.5.1                  | 1.0.0                | 1.0.0
0.30.0                                                                    | 2.28.0           | 0.30.0      | 2.0.0   | 1.15.0 / 2.4.0    | 0.30.0                     | 0.30.0              | 0.30.0                    | 2.4.0                  | 0.30.0               | 0.30.0
0.29.0                                                                    | 2.28.0           | 0.29.0      | 2.0.0   | 1.15.0 / 2.4.0    | 0.29.0                     | 0.29.0              | 0.29.0                    | 2.4.0                  | 0.29.0               | 0.29.0
0.28.0                                                                    | 2.28.0           | 0.28.0      | 2.0.0   | 1.15.0 / 2.4.0    | 0.28.0                     | 0.28.0              | 0.28.0                    | 2.4.0                  | 0.28.0               | 0.28.1
0.27.0                                                                    | 2.27.0           | 0.27.0      | 2.0.0   | 1.15.0 / 2.4.0    | 0.27.0                     | 0.27.0              | 0.27.0                    | 2.4.0                  | 0.27.0               | 0.27.0
0.26.4                                                                    | 2.28.0           | 0.26.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.26.1                     | 0.26.0              | 0.26.0                    | 2.3.0                  | 0.26.0               | 0.26.0
0.26.3                                                                    | 2.25.0           | 0.26.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.26.0                     | 0.26.0              | 0.26.0                    | 2.3.0                  | 0.26.0               | 0.26.0
0.26.1                                                                    | 2.25.0           | 0.26.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.26.0                     | 0.26.0              | 0.26.0                    | 2.3.0                  | 0.26.0               | 0.26.0
0.26.0                                                                    | 2.25.0           | 0.26.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.26.0                     | 0.26.0              | 0.26.0                    | 2.3.0                  | 0.26.0               | 0.26.0
0.25.0                                                                    | 2.25.0           | 0.24.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.25.0                     | 0.25.0              | 0.25.0                    | 2.3.0                  | 0.25.0               | 0.25.0
0.24.1                                                                    | 2.24.0           | 0.24.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.24.1                     | 0.24.0              | 0.24.3                    | 2.3.0                  | 0.24.1               | 0.24.1
0.24.0                                                                    | 2.24.0           | 0.24.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.24.1                     | 0.24.0              | 0.24.3                    | 2.3.0                  | 0.24.1               | 0.24.1
0.23.1                                                                    | 2.24.0           | 0.23.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.23.1                     | 0.23.0              | 0.23.0                    | 2.3.0                  | 0.23.0               | 0.23.0
0.23.0                                                                    | 2.23.0           | 0.23.0      | 0.17.0  | 1.15.0 / 2.3.0    | 0.23.0                     | 0.23.0              | 0.23.0                    | 2.3.0                  | 0.23.0               | 0.23.0
0.22.2                                                                    | 2.21.0           | 0.22.1      | 0.16.0  | 1.15.0 / 2.2.0    | 0.22.2                     | 0.22.2              | 0.22.2                    | 2.2.0                  | 0.22.0               | 0.22.1
0.22.1                                                                    | 2.21.0           | 0.22.1      | 0.16.0  | 1.15.0 / 2.2.0    | 0.22.2                     | 0.22.2              | 0.22.2                    | 2.2.0                  | 0.22.0               | 0.22.1
0.22.0                                                                    | 2.21.0           | 0.22.0      | 0.16.0  | 1.15.0 / 2.2.0    | 0.22.0                     | 0.22.0              | 0.22.1                    | 2.2.0                  | 0.22.0               | 0.22.0
0.21.5                                                                    | 2.17.0           | 0.21.2      | 0.15.0  | 1.15.0 / 2.1.0    | 0.21.5                     | 0.21.1              | 0.21.5                    | 2.1.0                  | 0.21.2               | 0.21.4
0.21.4                                                                    | 2.17.0           | 0.21.2      | 0.15.0  | 1.15.0 / 2.1.0    | 0.21.5                     | 0.21.1              | 0.21.5                    | 2.1.0                  | 0.21.2               | 0.21.4
0.21.3                                                                    | 2.17.0           | 0.21.2      | 0.15.0  | 1.15.0 / 2.1.0    | 0.21.5                     | 0.21.1              | 0.21.5                    | 2.1.0                  | 0.21.2               | 0.21.4
0.21.2                                                                    | 2.17.0           | 0.21.2      | 0.15.0  | 1.15.0 / 2.1.0    | 0.21.5                     | 0.21.1              | 0.21.5                    | 2.1.0                  | 0.21.2               | 0.21.4
0.21.1                                                                    | 2.17.0           | 0.21.2      | 0.15.0  | 1.15.0 / 2.1.0    | 0.21.4                     | 0.21.1              | 0.21.4                    | 2.1.0                  | 0.21.2               | 0.21.3
0.21.0                                                                    | 2.17.0           | 0.21.0      | 0.15.0  | 1.15.0 / 2.1.0    | 0.21.0                     | 0.21.0              | 0.21.1                    | 2.1.0                  | 0.21.0               | 0.21.0
0.15.0                                                                    | 2.16.0           | 0.15.0      | 0.15.0  | 1.15.0            | 0.15.0                     | 0.15.0              | 0.15.2                    | 1.15.0                 | 0.15.0               | 0.15.1
0.14.0                                                                    | 2.14.0           | 0.14.0      | 0.14.0  | 1.14.0            | 0.14.1                     | 0.14.0              | 0.14.0                    | 1.14.0                 | 0.14.0               | n/a
0.13.0                                                                    | 2.12.0           | 0.13.2      | n/a     | 1.13.1            | 0.13.1                     | 0.13.0              | 0.13.2                    | 1.13.0                 | 0.13.0               | n/a
0.12.0                                                                    | 2.10.0           | 0.13.2      | n/a     | 1.12.0            | 0.12.0                     | 0.12.1              | 0.12.1                    | 1.12.0                 | 0.12.0               | n/a

## Resources

*   [TFX tutorials ](https://www.tensorflow.org/tfx/tutorials)
*   [TensorFlow Extended (YouTube)](https://www.youtube.com/playlist?list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F)
*   [ MLOps Specialization ](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)
*   [ML Pipelines on Google Cloud](https://www.coursera.org/learn/ml-pipelines-google-cloud?specialization=preparing-for-google-cloud-machine-learning-engineer-professional-certificate)
*   [Manage a production ML pipeline with TFX](https://www.youtube.com/watch?v=QQ13-Tkrbls)
*   [How to build an ML pipeline with TFX](https://www.youtube.com/watch?v=17l3VR2MIeg)
