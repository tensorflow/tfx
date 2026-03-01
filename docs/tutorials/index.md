# Tensorflow in Production Tutorials

These tutorials will get you started, and help you learn a few different ways of
working with TFX for production workflows and deployments.  In particular,
you'll learn the two main styles of developing a TFX pipeline:

* Using the `InteractiveContext` to develop a pipeline in a notebook, working
  with one component at a time.  This style makes development easier and more
  Pythonic.
* Defining an entire pipeline and executing it with a runner.  This is what your
  pipelines will look like when you deploy them.

## Getting Started Tutorials

<div class="grid cards" markdown>

-   __1. Starter Pipeline__

    ---

    Probably the simplest pipeline you can build, to help you get started. Click
    the _Run in Google&nbsp;Colab_ button.

    [:octicons-arrow-right-24: Starter Pipeline](tfx/penguin_simple)

-   __2. Adding Data Validation__

    ---

    Building on the simple pipeline to add data validation components.

    [:octicons-arrow-right-24: Data Validation](tfx/penguin_tfdv)

-   __3. Adding Feature Engineering__

    ---

    Building on the data validation pipeline to add a feature engineering component.

    [:octicons-arrow-right-24: Feature Engineering](tfx/penguin_tft)

-   __4. Adding Model Analysis__

    ---

    Building on the simple pipeline to add a model analysis component.

    [:octicons-arrow-right-24: Model Analysis](tfx/penguin_tfma)

</div>


## TFX on Google Cloud

Google Cloud provides various products like BigQuery, Vertex AI to make your ML
workflow cost-effective and scalable. You will learn how to use those products
in your TFX pipeline.

<div class="grid cards" markdown>

-   __Running on Vertex Pipelines__

    ---

    Running pipelines on a managed pipeline service, Vertex Pipelines.

    [:octicons-arrow-right-24: Vertex Pipelines](tfx/gcp/vertex_pipelines_simple)

-   __Read data from BigQuery__

    ---

    Using BigQuery as a data source of ML pipelines.

    [:octicons-arrow-right-24: BigQuery](tfx/gcp/vertex_pipelines_bq)

-   __Vertex AI Training and Serving__

    ---

    Using cloud resources for ML training and serving with Vertex AI.

    [:octicons-arrow-right-24: Vertex Training and Serving](tfx/gcp/vertex_pipelines_vertex_training)

-   __TFX on Cloud AI Platform Pipelines__

    ---

    An introduction to using TFX and Cloud AI Platform Pipelines.

    [:octicons-arrow-right-24: Cloud Pipelines](tfx/cloud-ai-platform-pipelines)

</div>

## Next Steps

Once you have a basic understanding of TFX, check these additional tutorials and
guides. And don't forget to read the [TFX User Guide](../../guide).

<div class="grid cards" markdown>

-   __Complete Pipeline Tutorial__

    ---

    A component-by-component introduction to TFX, including the _interactive
    context_, a very useful development tool. Click the _Run in
    Google&nbsp;Colab_ button.

    [:octicons-arrow-right-24: Keras](tfx/components_keras)

-   __Custom Component Tutorial__

    ---

    A tutorial showing how to develop your own custom TFX components.

    [:octicons-arrow-right-24: Custom Component](tfx/python_function_component)

-   __Data Validation__

    ---

    This Google&nbsp;Colab notebook demonstrates how TensorFlow Data Validation
    (TFDV) can be used to investigate and visualize a dataset, including
    generating descriptive statistics, inferring a schema, and finding
    anomalies.

    [:octicons-arrow-right-24: Data Validation](data_validation/tfdv_basic)

-   __Model Analysis__

    ---

    This Google&nbsp;Colab notebook demonstrates how TensorFlow Model Analysis
    (TFMA) can be used to investigate and visualize the characteristics of a
    dataset and evaluate the performance of a model along several axes of
    accuracy.

    [:octicons-arrow-right-24: Model Analysis](model_analysis/tfma_basic)

-   __Serve a Model__

    ---

    This tutorial demonstrates how TensorFlow Serving can be used to serve a
    model using a simple REST API.

    [:octicons-arrow-right-24: Model Analysis](serving/rest_simple)

</div>

## Videos and Updates

Subscribe to the [TFX YouTube
Playlist](https://www.youtube.com/playlist?list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F)
and [blog](https://blog.tensorflow.org/search?label=TFX&max-results=20) for the
latest videos and updates.


- [TFX: Production ML with TensorFlow in 2020](https://youtu.be/I3MjuFGmJrg)

<div class="video-wrapper"><iframe width="240" src="https://www.youtube.com/embed/I3MjuFGmJrg" frameborder="0" allowfullscreen></iframe></div>

- [TFX: Production ML pipelines with TensorFlow](https://youtu.be/TA5kbFgeUlk)

<div class="video-wrapper"><iframe width="240" src="https://www.youtube.com/embed/TA5kbFgeUlk" frameborder="0" allowfullscreen></iframe></div>

- [Taking Machine Learning from Research to Production](https://youtu.be/rly7DqCbtKw)

<div class="video-wrapper"><iframe width="240" src="https://www.youtube.com/embed/rly7DqCbtKw" frameborder="0" allowfullscreen></iframe></div>
