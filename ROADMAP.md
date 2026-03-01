### TFX OSS roadmap
This highlights the main OSS efforts for the TFX team, along with the history.
If you're interested in contributing in one of these areas,
[contributions](https://github.com/tensorflow/tfx/blob/master/CONTRIBUTING.md)
are always welcome, especially in areas that extend TFX into infrastructure
currently not widely in use at Google.

#### _Vision_
*   Democratize access to machine learning (ML) best practices, tools, and code.
*   Enable users to easily run production ML pipelines locally, on public
    clouds, on premises, and in heterogeneous computing environments.

#### _Goals_
*   **Help the community:** Help enterprises realize large-scale production ML
capabilities similar to what we have available at Google.  We recognize that
every enterprise has unique infrastructure challenges, and we want TFX to be
open and adaptable to those challenges.
*   **Stimulate innovation:** Machine learning is a rapid, innovative field and
we want TFX to help researchers and engineers both realize and contribute to
that innovation.  Likewise, we want TFX to be interoperable with other ML
efforts in the open source community.
*   **Usability:** We want the experience of developing and deploying a model
in production to be as frictionless as possible throughout the entire journey --
from the initial efforts of building a model to the final touches of deploying
in production.

#### _Specific efforts underway_

##### Extensibility
*   Participate in and extend support for other OSS efforts, initially:
[Apache Beam](https://beam.apache.org/),
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd),
[Tensorboard](https://www.tensorflow.org/tensorboard),
[Kubeflow](https://www.kubeflow.org/), and
[TensorFlow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/).
*   Make TFX more ML framework neutral to enable wider usage.
*   Encourage the discovery and reuse of external contributions, TFX-Addons.
*   Extend portability across additional cluster computing frameworks.

##### Usability
*   Improving the testing capabilities for OSS developers.
*   Support for more advanced orchestration semantics.
*   Create examples and templates for more ML verticals.

##### Performance
*   Better TPU support on Cloud.
*   Support for more performant file storage formats than TFRecords.
*   Better telemetry for users to understand the behavior of components in a
    TFX pipeline.

##### Innovation and collaboration
*   Formalize Special Interest Groups (SIGs) for specific aspects of TFX to
    accelerate community innovation and collaboration.
*   Early access to new features.

#### History

[Towards ML Engineering: A Brief History Of TensorFlow Extended (TFX)](https://blog.tensorflow.org/2020/09/brief-history-of-tensorflow-extended-tfx.html)

*   Q2 2022
    *   Dynamic Exec Properties support for Vertex.
    *   Vertex Machine type configuration.
    *   Investige other splitable file storage formats than just TFRecord.
*   Q3 2020
    *   Component Launches & Enhancements.
    *   Cloud AI Platform integration with BulkInferrer.
    *   Multi Framework Support in TFX Components.
    *   On Device Support for TFJS in Evaluator component.
    *   Intermediate Representation (IR).
*   Q2 2020
    *   Custom component authoring was made easier by supporting python function
        and custom container.
    *   Created a new TFJS rewriter.
    *   Created a new InfraValidator component.
    *   Created a new Tuner component.
    *   Introduced artifact types for primitive values and generic type.
*   Q1 2020
    *   Released support for native Keras.
    *   Released initial integration with tf.Lite.
    *   New template to create pipelines for on-premise and cloud.
    *   New ComponentSpec and standard artifact types published.
    *   Allow pipelines to be parameterized with `RuntimeParameters`.
    *   Enabled warm-starting for estimator based trainers.
*   Q4 2019
    *   Added limited support for TF.Keras through
        `tf.keras.estimator.model_to_estimator()`.
*   Q3 2019
    *   Support for local orchestrator through Apache Beam.
    *   Experimental support for interactive development on Jupyter notebook.
    *   Experimental support for TFX CLI released.
    *   Started to publish public [RFCs](
        https://github.com/tensorflow/community/tree/master/rfcs)
        to the tensorflow/community project. This will be an ongoing effort.
*   Q2 2019
    *   Support for Python3.
    *   Support for Apache Spark and Apache Flink runners (with examples).
    *   Custom executors (with examples).
*   Q1 2019
    *   [TFX](https://www.tensorflow.org/tfx/guide) end-to-end pipeline,
config, and orchestration initial release.
    *   [ml.metadata](https://www.tensorflow.org/tfx/guide/mlmd) initial
release.
*   Q3 2018
    *   [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv)
initial release.
*   Q1 2018
    *   [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma)
initial release.
*   Q1 2017
    *   [TensorFlow Transform](https://www.tensorflow.org/tfx/guide/tft)
initial release.
*   Q1 2016
    *   [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
initial release.
