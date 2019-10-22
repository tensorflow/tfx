### TensorFlow Extended Open-Source Software Roadmap
This highlights the main OSS efforts of the TFX team in 2019 and H1 2020. If
you're interested in contributing to any of the areas listed below, check out the
[contributions](https://github.com/tensorflow/tfx/blob/master/CONTRIBUTING.md) doc.
Contributions are always welcome, especially in areas that extend TFX into infrastructure
that's currently not widely in use at Google

#### _Vision_
*   Democratize access to machine learning (ML) best practices, tools, and code
*   Enable users to easily run production ML pipelines on public clouds, on-prem infrastructure
and in heterogeneous computing environments

#### _Goals_
*   Help enterprises realize large-scale production ML capabilities similar to
what we have available at Google.  We recognize that every enterprise has unique
infrastructure challenges and we want TFX to be open and adaptable to those
challenges
*   Stimulate innovation: Machine learning is a rapid, innovative field and we
want TFX to help researchers and engineers realize and contribute to that
innovation.  Likewise, we want TFX to be interoperable with other ML efforts in
the open source community
*   Usability: We want the journey to deploy a model in production to be as
frictionless as possible throughout the entire journey. From the initial
efforts of building a model to the final touches of deploying one in production

#### _Specific efforts underway_

##### Extensibility
*   Encourage the discovery and reuse of external contributions
*   Participate in and extend support for other OSS efforts, namely:
[Apache Beam](https://beam.apache.org/),
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd),
[Kubeflow](https://www.kubeflow.org/),
[TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard), and
[TensorFlow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/)
*   Extend portability across additional cluster-computing frameworks,
orchestrators, and data representations

##### Performance
*   Better distributed training support
(See [DistributionStrategy](https://www.tensorflow.org/guide/distribute_strategy))
*   Better telemetry for users to understand the behavior of components in a
TFX pipeline

##### Usability
*   Support of TensorFlow 2.0 in two phases:
    1.  The first phase will provide the following:
        Existing TFX pipelines can continue to use TensorFlow 1.X. To switch to
        TensorFlow 2.X, see the [TensorFlow migration guide](
        https://www.tensorflow.org/guide/migrate)
        New TFX pipelines should use Keras (via
        `tf.keras.estimator.model_to_estimator()`) and TensorFlow 2.X
    1.  The second phase will enable the remainder of TensorFlow 2.X
        functionality, including tf.distribute and Keras without Estimator
*   Integration with TensorBoard and TF Hub/AI Hub
*   Improving the testing capabilities for OSS developers
*   Increased interoperability with Kubeflow Pipelines
*   Support for training on continuously arriving data

##### Education
*   More pipeline code examples, including DIY orchestrators and custom
components

##### Innovation and collaboration
*   Formalize Special Interest Groups (SIGs) for specific aspects of TFX to
accelerate community innovation and collaboration
*   Early access to new features

#### History
*   Q3 2019: Support for local orchestrators through Apache Beam
*   Q3 2019: Experimental support for interactive development on Jupyter notebook
*   Q3 2019: Experimental support for TFX CLI released
*   Q3 2019: Multiple public [RFCs](https://github.com/tensorflow/community/tree/master/rfcs) published to the TensorFlow/community project
*   Q2 2019: Support for Python3
*   Q2 2019: Apache Spark and Apache Flink runners (with examples)
*   Q2 2019: Custom executors (with examples)
*   Q1 2019: [TFX](https://www.tensorflow.org/tfx/guide) end-to-end pipeline,
config, and orchestration initial release
*   Q1 2019: [ml.metadata](https://www.tensorflow.org/tfx/guide/mlmd) initial
release
*   Q3 2018: [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv)
initial release
*   Q1 2018: [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma)
initial release
*   Q1 2017: [TensorFlow Transform](https://www.tensorflow.org/tfx/guide/tft)
initial release
*   Q1 2016: [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
initial release
