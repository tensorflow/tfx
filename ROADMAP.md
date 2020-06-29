### TFX OSS roadmap
This highlights the main OSS efforts for the TFX team in 2019 and H2 2020. If
you're interested in contributing in one of these areas,
[contributions](https://github.com/tensorflow/tfx/blob/master/CONTRIBUTING.md)
are always welcome, especially in areas that extend TFX into infrastructure
currently not widely in use at Google.

#### _Vision_
*   Democratize access to machine learning (ML) best practices, tools, and code.
*   Enable users to easily run production ML pipelines locally, on public
    clouds, on premises, and in heterogeneous computing environments.

#### _Goals_
*   Help enterprises realize large-scale production ML capabilities similar to
what we have available at Google.  We recognize that every enterprise has unique
infrastructure challenges, and we want TFX to be open and adaptable to those
challenges.
*   Stimulate innovation: Machine learning is a rapid, innovative field and we
want TFX to help researchers and engineers both realize and contribute to that
innovation.  Likewise, we want TFX to be interoperable with other ML efforts in
the open source community.
*   Usability: We want the experience to deploy a model in production to be as
frictionless as possible throughout the entire journey -- from the initial
efforts building a model to the final touches of deploying in production.

#### _Specific efforts underway_

##### Extensibility
*   Encourage the discovery and reuse of external contributions.
*   Participate in and extend support for other OSS efforts, initially:
[Apache Beam](https://beam.apache.org/),
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd),
[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), and
[TensorFlow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/).
*   Make TFX more ML framework neutral to enable wider usage.
*   Extend portability across additional cluster computing frameworks and data
representations.

##### Performance
*   Better distributed training support
([DistributionStrategy](https://www.tensorflow.org/guide/distribute_strategy)).
*   Support for more performant file storage formats than TFRecords.
*   Better telemetry for users to understand the behavior of components in a
TFX pipeline.

##### Usability
*   Support more distributed strategies in TensorFlow 2.x.
*   Improving the testing capabilities for OSS developers.
*   Reach feature parity and make it easy to move ML focused pipelines from
    Kubeflow pipelines (KFP) to TFX DSL. Also share the same pipeline
    intermediate representation for both platforms to guarantee semantics and
    data model consistency.
*   Support for training on continuously arriving data and more advanced
    orchestration semantics.
*   Create examples and templates for more ML verticals.

##### Education
*   Work with [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) to
    publish standard ontology types and show case them through TFX.

##### Innovation and collaboration
*   Further enhance integration with tf.Lite to better support mobile and edge
devices.
*   Formalize Special Interest Groups (SIGs) for specific aspects of TFX to
accelerate community innovation and collaboration.
*   Early access to new features.

#### History
*   Q2 2020
    *   Custom component authoring was made easier by supporting python function
        and custom container.
    *   Created a new TFJS rewriter.
    *   Created a new infra validation component.
    *   Introduced artifact types for primitive values and generic type.
*   Q1 2020
    *   Released support for native Keras in TFX.
    *   Released initial integration with tf.Lite.
    *   New template to create pipelines in TFX for on-premise and cloud.
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
