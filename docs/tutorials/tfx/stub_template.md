## Testing the pipeline using Stub Executors

### Introduction

**You should complete
[template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb)
tutorial up to *Step 6* in order to proceed this tutorial.**

This document will provide instructions to test a TensorFlow Extended (TFX)
pipeline using `BaseStubExecuctor`, which generates fake artifacts using the
golden test data. This is intended for users to replace executors they don't
want to test so that they could save time from running the actual executors.
Stub executor is provided with TFX Python package under
`tfx.experimental.pipeline_testing.base_stub_executor`.

This tutorial serves as an extension to `template.ipynb` tutorial, thus you will
also use
[Taxi Trips dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago. We strongly encourage you to try modifying the
components prior to utilizing stub executors.

### 1. Record the pipeline outputs in Google Cloud Storage

We first need to record the pipeline outputs so that the stub executors can copy
over the artifacts from the recorded outputs.

Since this tutorial assumes that you have completed `template.ipynb` up to step
6, a successful pipeline run must have been saved in the
[MLMD](https://www.tensorflow.org/tfx/guide/mlmd). The execution information in
MLMD can be accessed using gRPC server.

Open a Terminal and run the following commands:

1.  Generate a kubeconfig file with appropriate credentials: `bash gcloud
    container clusters get-credentials $cluster_name --zone $compute_zone
    --project $gcp_project_id` `$compute_zone` is region for gcp engine and
    `$gcp_project_id` is project id of your GCP project.

2.  Set up port-forwarding for connecting to MLMD: `bash nohup kubectl
    port-forward deployment/metadata-grpc-deployment -n $namespace $port:8080 &`
    `$namespace` is the cluster namespace and `$port` is any unused port that
    will be used for port-forwarding.

3.  Clone the tfx GitHub repository. Inside the tfx directory, run the following
    command:

```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<gcp_project_id>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

`$output_dir` should be set to a path in Google Cloud Storage where the pipeline
outputs are to be recorded, so make sure to replace `<gcp_project_id>` with GCP
project id.

`$host` and `$port` are hostname and port of the metadata grpc server to connect
to MLMD. `$port` should be set to the port number you used for port-forwarding
and you may set "localhost" for the hostname.

In `template.ipynb` tutorial, the pipeline name is set as "my_pipeline" by
default, so set `pipeline_name="my_pipeline"`. If you have modified the pipeline
name when running the template tutorial, you should modify the `--pipeline_name`
accordingly.

### 2. Enable Stub Executors in Kubeflow DAG Runner

First, make sure that the predefined template has been copied to your project
directory using `tfx template copy` CLI command. It is necessary to edit the
following two files in the copied source files.

1.  Create a file called `stub_component_launcher.py` in the directory where
    kubeflow_dag_runner.py is located, and put following content to it.

    ```python
    from tfx.experimental.pipeline_testing import base_stub_component_launcher
    from pipeline import configs

    class StubComponentLauncher(
        base_stub_component_launcher.BaseStubComponentLauncher):
      pass

    # GCS directory where KFP outputs are recorded
    test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO: customize self.test_component_ids to test components, replacing other
    # component executors with a BaseStubExecutor.
    test_component_ids = ['Trainer']
    StubComponentLauncher.initialize(
        test_data_dir=test_data_dir,
        test_component_ids=test_component_ids)
    ```

    NOTE: This stub component launcher cannot be defined within
    `kubeflow_dag_runner.py` because launcher class is imported by the module
    path.

1.  Set component ids to be list of component ids that are to be tested (in
    other words, other components' executors are replaced with BaseStubExecutor)
    .

1.  Open `kubeflow_dag_runner.py`. Add following import statement at the top to
    use `StubComponentLauncher` class we just added.

    ```python
    import stub_component_launcher
    ```

1.  In `kubeflow_dag_runner.py`, add `StubComponentLauncher` class to
    `KubeflowDagRunnerConfig`'s `supported_launcher_class` to enable launch of
    stub executors:

    ```python
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        supported_launcher_classes=[
            stub_component_launcher.StubComponentLauncher
        ],
    ```

### 3. Update and run the pipeline with stub executors

Update the existing pipeline with modified pipeline definition with stub
executors.

```bash
tfx pipeline update --pipeline-path=kubeflow_dag_runner.py \
  --endpoint=$endpoint --engine=kubeflow
```

`$endpoint` should be set to your KFP cluster endpoint.

Run the following command to create a new execution run of your updated
pipeline.

```bash
tfx run create --pipeline-name $pipeline_name --endpoint=$endpoint \
  --engine=kubeflow
```

## Cleaning up

Use command `fg` to access the port-forwarding in the background then ctrl-C to
terminate. You can delete the directory with recorded pipeline outputs using
`gsutil -m rm -R $output_dir`.

To clean up all Google Cloud resources used in this project, you can
[delete the Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects)
you used for the tutorial.

Alternatively, you can clean up individual resources by visiting each
consoles: - [Google Cloud Storage](https://console.cloud.google.com/storage) -
[Google Container Registry](https://console.cloud.google.com/gcr) -
[Google Kubernetes Engine](https://console.cloud.google.com/kubernetes)
