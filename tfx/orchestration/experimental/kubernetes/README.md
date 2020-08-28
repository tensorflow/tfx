# TFX Orchestration on Kubernetes

This orchestrator is experimental and is not suitable for production use. For
pipeline deployment on Kubernetes, we currently recommend that you use the
Kubeflow Pipelines orchestrator found in `tfx/orchestration/kubeflow`

This package provides experimental support for executing synchronous TFX
pipelines in an on premise Kubernetes cluster as an alternative to
[KubeFlow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/)
. Use the workflow below to set up your cluster for pipeline execution.

## Step 1: Set up a Kubernetes cluster

### Kubernetes setup

To create your own on-premise or cloud-based Kubernetes cluster, follow the
[Kubernetes Getting Started Guide](https://kubernetes.io/docs/setup/) to set up
your Kubernetes environment.

### Creating a Google Kubernetes Engine cluster on Google Cloud Platform

If you would like to run a managed Kubernetes cluster on Google Cloud, follow
the
[Google Kubernetes Engine Quickstart Guide](https://cloud.google.com/kubernetes-engine/docs/quickstart).

## Step 2: Set up Jupyter Notebook Service and MySQL MLMD

First, ensure that you are in the base TFX directory. Use the following command
to deploy the default Jupyter Notebook and MySQL resources: `kubectl apply -k
tfx/orchestration/experimental/kubernetes/yaml/` **Important: If you are using a
Kubernetes cluster other than GKE, go to
tfx/orchestration/experimental/kubernetes/yaml/mysql-pv.yaml and follow the
instructions to modify the configurations for your cluster.**

### Using the In-Cluster Jupyter Notebook

The in-cluster Jupyter Notebook allows you to edit files and run pipelines
directly from within your Kubernetes cluster. Note that the contents of this
notebook server are ephemeral, so we suggest using this for testing only.

To log on to your Jupyter server, you need the log in token. You may customize a
log in password after the first time you log in. To obtain the log in token,
first use `kubectl get pods` to locate the pod name starting with "jupyter-".
Then, read the pod start-up log to obtain the login password by replacing
$YOUR_POD_NAME with the name of the jupyter pod: `kubectl logs $YOUR_POD_NAME`

Finally, you may use port forwarding to access the server at `localhost:8888`:
`kubectl port-forward $YOUR_POD_NAME 8888:8888`

### Using the MySQL MLMD

The MySQL Service will be used as a
[metadata store](https://www.tensorflow.org/tfx/guide/mlmd) for your TFX
pipelines. You do not need to interact with it by default, but it may be useful
for debugging pipeline executions.

To access the service from the command line, use: `kubectl run -it --rm
--image=mysql:5.6 --restart=Never mysql-client -- mysql --host mysql`

To use the MySQL instance as a metadata store in your TFX pipeline or
interactive context, first create a custom metadata connection config:
`_metadata_connection_config = metadata.mysql_metadata_connection_config(
host='mysql', port=3306, username='root', database='mysql', password='')`

Now, you can use this in your pipeline by passing it into the constructor for
`pipeline.Pipeline`: `pipeline.Pipeline( pipeline_name=pipeline_name,
pipeline_root=pipeline_root, components=[ # ... ],
metadata_connection_config=_metadata_connection_config,
beam_pipeline_args=beam_pipeline_args)`

Similarly, you can initialize a custom interactive context to use this metadata
store with: `context =
InteractiveContext(metadata_connection_config=_metadata_connection_config)`

## Step 3: Build and upload your TFX image

The default container image used for executing TFX pipeline components is
`tensorflow/tfx`. If you would like to use a custom container image, you can
start by creating and a custom Dockerfile, for example: `FROM python:3.7 RUN pip
install tfx # Add your dependencies here.`

Once you have created your Dockerfile, you can build it while tagging your image
name: `docker build -t $YOUR_IMAGE_NAME .`

Then, upload the image to your cloud container registry: `docker push
$YOUR_IMAGE_NAME`
