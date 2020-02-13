### TFX OSS roadmap
This highlights the main OSS efforts for the TFX team in 2019 and H1 2020. If
you're interested in contributing in one of these areas,
[contributions](https://github.com/tensorflow/tfx/blob/master/CONTRIBUTING.md)
are always welcome, especially in areas that extend TFX into infrastructure
currently not widely in use at Google.

#### _Vision_
*   Democratize access to machine learning (ML) best practices, tools, and code.
*   Enable users to easily run production ML pipelines on public clouds, on
premises, and in heterogeneous computing environments.

#### _Goals_
*   Help enterprises realize large-scale production ML capabilities similar to
what we have available at Google.  We recognize that every enterprise has unique
infrastructure challenges, and we want TFX to be open and adaptable to those
challenges.
*   Stimulate innovation: Machine learning is a rapid, innovative field and we
want TFX to help researchers and engineers both realize and contribute to that
innovation.  Likewise, we want TFX to be interoperable with other ML efforts in
the open source community.
*   Usability: We want the journey to deploy a model in production to be as
frictionless as possible throughout the entire journey -- from the initial
efforts building a model to the final touches of deploying in production.

#### _Specific efforts underway_

##### Extensibility
*   Encourage the discovery and reuse of external contributions.
*   Participate in and extend support for other OSS efforts, initially:
[Apache Beam](https://beam.apache.org/),
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd),
[Kubeflow](https://www.kubeflow.org/),
[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), and
[TensorFlow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/).
*   Align ML framework support with Kubeflow pipelines.
*   Extend portability across additional cluster computing frameworks,
orchestrators, and data representations.

##### Performance
*   Better distributed training support
([DistributionStrategy](https://www.tensorflow.org/guide/distribute_strategy)).
*   Better telemetry for users to understand the behavior of components in a
TFX pipeline.

##### Usability
*   Complete the support for tensorflow 2.x functionaties, including
    tf.distribute and Keras without Estimator.
*   Improving the testing capabilities for OSS developers.
*   Increased interoperability with Kubeflow Pipelines, with a focus on
    providing more flexibility from unified DSL and converging on pipeline
    presentation and orchestration semantics.
*   Support for training on continuously arriving data and more advanced
    orchestration semantics.

##### Education
*   New template in TFX OSS to ease creation of TFX pipelines.
*   More pipeline code examples, including DIY orchestrators and custom
components.
*   Work with [ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd) to
    publish standard ontology types and show case them through TFX.

##### Innovation and collaboration
*   Support mobile and edge devices by integrating with tf.lite.
*   Formalize Special Interest Groups (SIGs) for specific aspects of TFX to
accelerate community innovation and collaboration.
*   Early access to new features.

#### History
*   Q1 2020
    *   New ComponentSpec and standard artifact types published.
    *   Allow pipelines to be parameterized with `RuntimeParameters`.
    *   Enabled warm-starting for estimator based trainers.
*   Q4 2019
    *   Added limited support for TF.Keras through `tf.keras.estimator.model_to_estimator()`.
*   Q3 2019
    *   Support for local orchestrator through Apache Beam.
    *   Experimental support for interactive development on Jupyter notebook.
    *   Experimental support for TFX CLI released.
    *   Multiple public [RFCs](https://github.com/tensorflow/community/tree/master/rfcs) published to the tensorflow/community project.
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
