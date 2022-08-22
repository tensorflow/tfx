# TFX Centralized Kubernetes Orchestrator

Disclaimer: This orchestrator is experimental and we don't have any plans to
support this officially in production, as of July 2022.

![image](https://user-images.githubusercontent.com/57027695/184351225-3e9c916b-ebaa-4d85-93a5-a9e7e924d747.png)

This package aims to provide a centralized orchestrator on kubernetes, without
relying on external orchestration tools such as
[KubeFlow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/).
To try it out, please follow the steps below.

# Setup

Follow these step if you are running the orchestrator for the first time.

## Step 1: Set up a Kubernetes cluster

Refer to
[this link](https://github.com/tensorflow/tfx/tree/master/tfx/orchestration/experimental/kubernetes#step-1-set-up-a-kubernetes-cluster)
for set up.

## Step 2: Build a new docker image

Current official tfx image doesn't support this orchestrator, as `entrypoint.py`
is not included in the image. Thus, you need to build a new image before trying
out examples below.

To fully utilize the features in the orchestrator, you should build your own
image which includes your code on the components you would like to run.

Under the root directory of github checkout, run `export
DOCKER_IMAGE_REPO=gcr.io/{your_GKE_project_name}/{image_name}
TFX_DEPENDENCY_SELECTOR=NIGHTLY ./tfx/tools/docker/build_docker_image.sh docker
push ${DOCKER_IMAGE_REPO}` to build and push a docker image to your container.

Then, change the `tfx_image` parameter of
`kubernetes_job_runner.KubernetesJobRunner` (line 90 of
kubernetes_task_scheduler.py) to the name of your image.

TODO(b/240237394): Read the image information from the platform config.

## Step 3: Set up MySQL MLMD

After checking that you are inside the base TFX directory, use the following
command to deploy the MySQL resources: `kubectl apply -f
tfx/orchestration/experimental/kubernetes/yaml/mysql-pv.yaml kubectl apply -f
tfx/orchestration/experimental/kubernetes/yaml/mysql.yaml`

## Step 4: Create MySQL Database

Next, you need to create a database you would use for MLMD. Creating a database
locally using port-fowarding is recommended.

Run `kubectl port-forward {mysql_pod_name} {your_port}:3306` and in a separate
terminal, run `mysql -h localhost -P {your_port} -u root` to make MySQL
connection.

Create database by `CREATE DATABASE {database_name};`

# How to Use

## Running a sample pipeline.

1) Run main.py with necessary flags, which serves as the orchestration loop.

Orchestrator loop runs outside the kubernetes cluster for the current
implementation. Thus, while port-forwarding with above command, run `main.py`
with necessary flags as shown below.

```
python tfx/orchestration/experimental/centralized_kubernetes_orchestrator/main.py
--mysql_port={your_port} --mysql_host={your_host} --mysql_username={your_username} --mysql_database={your_database_name}
```

If you are running using localhost, specify mysql_host as 127.0.0.1, not
localhost.

2) In a separate terminal, execute `run_sample_pipeline.py` with necessary
flags, as shown below.

Sample command: `python
tfx/orchestration/experimental/centralized_kubernetes_orchestrator/examples/run_sample_pipeline.py
--bucket={your_gcs_bucket_name}`
