# TFX Orchestration on Kubernetes

This package provides experimental support for executing synchronous TFX pipelines in an on premise Kubernetes cluster as an alternative to [KubeFlow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/). Use the workflow below to set up your cluster for pipeline execution.

## Step 1: setting up a GKE cluster

Set up a Kubernetes Cluster on Google Kubernetes Engine. Create a cluster under Clusters -> Create Cluster. **Important: under "Node Pools > default-pool > Security", select "Allow full access to all Cloud APIs".**  Without this, you will only have read-only access to cloud storage, and can't change this without deleting and recreating your cluster.

The following sections assume you have command line access to the cluster via kubectl. If you are using GKE, you can go to **Kubernetes Engine -> Clusters -> Your Cluster -> connect** and follow the instructions.


