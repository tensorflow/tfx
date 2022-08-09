# TFX Centralized Kubernetes Orchestrator

Code in this directory is under development.
This orchestrator is experimental and we don't have any plans to support this 
officially in production, as of July 2022.

# How to Use
TODO(b/240237394): Add more instructions to run examples.
1. Running a sample component. 
First, build a docker image 
```examples/run_sample_component.py``` showcases how to run ImportSchemaGen
component using the ```KubernetesJobRunner```. Please specify your GCS storage bucket,
docker image, job prefix, and container name as command line arguments.

Sample command: 
```
python tfx/orchestration/experimental/centralized_kubernetes_orchestrator/examples/run_sample_component.py docker_image=gcr.io/ysyang-flexing/nightly-tfx:latest job_prefix=sample-job container_name=centralized-orchestrator storage_bucket=ysyang-flexing-kubeflowpipelines-default
```