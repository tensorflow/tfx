# Orchestrating TFX Pipelines

## Vertex AI Pipelines

[Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
is a managed service in Google Cloud Platform which helps you to automate,
monitor, and govern your ML systems by orchestrating your ML workflow in a
managed, serverless manner.

It is recommended to use TFX to define ML pipelines for Vertex AI Pipelines, if
you use TensorFlow in an ML workflow that processes terabytes of structured data
or text data. See also
[the Vertex AI guide](https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline#sdk)

Try the
[TFX on Cloud tutorials](/tfx/tutorials/tfx/gcp/vertex_pipelines_simple) running
in Colab to learn how to use Vertex AI Pipelines with TFX.
