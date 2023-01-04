# Orchestrating TFX Pipelines

## Local Orchestrator

Local orchestrator is a simple orchestrator that is included in the TFX Python
package. It runs pipelines in the local environment in a single process. It
provides fast iterations for development and debugging, but it is not suitable for
large production workloads. Please use [Vertex Pipelines](/tfx/guide/vertex) or
[Kubeflow Pipelines](/tfx/guide/kubeflow) for production use cases.

Try the [TFX tutorials](/tfx/tutorials/tfx/penguin_simple) running in Colab to
learn how to use the local orchestrator.
