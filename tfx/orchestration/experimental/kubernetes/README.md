# TFX Orchestration on Kubernetes

This package provides experimental support for executing synchronous TFX pipelines in an on premise Kubernetes cluster as an alternative to [KubeFlow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/). Use the workflow below to set up your cluster for pipeline execution.

## Step 1: Set up a GKE cluster

Set up a Kubernetes Cluster on Google Kubernetes Engine. Create a cluster under Clusters -> Create Cluster. **Important: under "Node Pools > default-pool > Security", select "Allow full access to all Cloud APIs".**  Without this, you will only have read-only access to cloud storage, and can't change this without deleting and recreating your cluster.

The following sections assume you have command line access to the cluster via kubectl. If you are using GKE, you can go to **Kubernetes Engine -> Clusters -> Your Cluster -> connect** and follow the instructions.

## Step 2: Set up Jupyter Notebook Service and Mysql MLMD

First, ensure that you are in the base TFX directory. Use the following command to deploy the defualt Jupyter Notebook and MySql resources:
```
kubectl apply -k tfx/orchestration/experimental/kubernetes/yaml/
```
**Important: If you are using a Kubernetes cluster other than GKE, go to tfx/orchestration/experimental/kubernetes/yaml/mysql-pv.yaml and follow the instructions to modify the configurations for your cluster.**

### Using the In-Cluster Jupyter Notebook
The in-cluster Jupyter Notebook allows you to edit files and run pipelines directly from within your Kubernetes cluster. The default Jupyter Notebook resource uses a [Nodeport](https://cloud.google.com/kubernetes-engine/docs/how-to/exposing-apps#creating_a_service_of_type_nodeport) to expose its service. To log on to your jupyter server, you need the external ip, port and log in token. You may customize a log in password after the first time you log in.

To obtain the log in token, first use `kubectl get pods` to locate the pod name starting with "jupyter-". Then, read the pod start-up log to obtain the login password by replacing $YOUR_POD_NAME with the name of the jupyter pod:
```
kubectl logs $YOUR_POD_NAME
```

To obtain the port of the service, look for the Nodeport attribute when typing:
```
kubectl describe service jupyter
```
The port would by default be some number in the range 30000-32767.

You can use the external ip address of any node in your cluster. Type:
```
kubectl desribe nodes
```
and look for the EXTERNAL_IP of any of the nodes.

Finally, you should be able to access your server at http:// $EXTERNAL_IP : $NODE_PORT

### Using the MySQL MLMD
The Mysql Service will be used as a [metadata store](https://www.tensorflow.org/tfx/guide/mlmd) for your TFX pipelines. You do not need to intereract with it by default, but it may be useful for debugging pipeline executions.

To access the service from the command line, use:
```
kubectl run -it --rm --image=mysql:5.6 --restart=Never mysql-client -- mysql --host mysql
```

To use the mysql instance as a metadata store ([example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_interactive.ipynb)), you can initialize a custom interactive context with:

```
_metadata_connection_config = metadata.mysql_metadata_connection_config(
    host='mysql', port=3306, username='root', database='mysql', password='')
context = InteractiveContext(metadata_connection_config=_metadata_connection_config)
```

## Step 3: Build and upload your TFX image

The default container image used for executing TFX pipeline components is `tensorflow/tfx`. If you would like to use a custom container image, modify the Dockerfile located in this directory, and build the new image:

```
docker build -t $YOUR_IMAGE_NAME .
```

Then, upload the image to your cloud container registry:

```
docker push $YOUR_IMAGE_NAME .
```
