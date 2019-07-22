### TFX OSS roadmap
This highlights the main OSS efforts for the TFX team in 2019. If you're
interested in contributing in one of these areas,
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
*   Enable additional modularity and extensibility across TFX, including the
ability for users to inject callbacks for TFX executors, create custom
executors, components and pipelines.  Encourage the discovery and reuse of these
new contributions.
*   Participate in and extend support for other OSS efforts, initially:
[Apache Beam](https://beam.apache.org/),
[ML Metadata](https://www.tensorflow.org/tfx/guide/mlmd),
[Kubeflow](https://www.kubeflow.org/),
[Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), and
[TensorFlow 2.0](https://www.tensorflow.org/versions/r2.0/api_docs/).
*   Extend portability across additional cluster computing frameworks,
orchestrators, and data representations.

##### Performance
*   Better distributed training support
([DistributionStrategy](https://www.tensorflow.org/guide/distribute_strategy)).

##### Usability
*   Support of TensorFlow 2.0, starting with Keras.
*   Integration with TensorBoard and Jupyter notebooks.
*   A unified command line interface (CLI) for users to perform critical user
journeys in different environments.
*   Improving the testing capabilities for OSS developers.
*   Lightweight local orchestrator.
*   Increased interoperability with Kubeflow Pipelines.

##### Education
*   More pipeline code examples, including DIY orchestrators and custom
components.
*   Incorporate community feedback (RFCs) to the TFX design review process.

##### Innovation and collaboration
*   Formalize Special Interest Groups (SIGs) for specific aspects of TFX to
accelerate community innovation and collaboration.
*   Early access to new features.

#### History
*   Q2 2019: Support for python3; spark and flink runners (with examples);
and custom executors (with examples).
*   Q1 2019: [TFX](https://www.tensorflow.org/tfx/guide) end-to-end pipeline,
config, and orchestration initial release.
*   Q1 2019: [ml.metadata](https://www.tensorflow.org/tfx/guide/mlmd) initial
release.
*   Q3 2018: [TensorFlow Data Validation](https://www.tensorflow.org/tfx/guide/tfdv)
initial release.
*   Q1 2018: [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/guide/tfma)
initial release.
*   Q1 2017: [TensorFlow Transform](https://www.tensorflow.org/tfx/guide/tft)
initial release.
*   Q1 2016: [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
initial release.
