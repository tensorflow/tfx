# Orchestrating TFX Pipelines

## Kubeflow Pipelines

[Kubeflow](https://www.kubeflow.org/) is an open source ML platform dedicated to
making deployments of machine learning (ML) workflows on Kubernetes simple,
portable and scalable.
[Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/pipelines-overview/)
is part of the Kubeflow platform that enables composition and execution of
reproducible workflows on Kubeflow, integrated with experimentation and notebook
based experiences. Kubeflow Pipelines services on Kubernetes include the hosted
Metadata store, container based orchestration engine, notebook server, and UI to
help users develop, run, and manage complex ML pipelines at scale. The Kubeflow
Pipelines SDK allows for creation and sharing of components and composition and
of pipelines programmatically.

See the
[TFX example on Kubeflow Pipelines](https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)
for details on running TFX at scale on Google cloud.
