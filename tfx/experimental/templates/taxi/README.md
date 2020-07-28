# Chicago Taxi TFX pipeline template

Please see [TFX on Cloud AI Platform Pipelines](
https://www.tensorflow.org/tfx/tutorials/tfx/cloud-ai-platform-pipelines)
tutorial to learn how to use this template.

## The dataset

This template uses the [Taxi Trips dataset](
https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago in the initial state, but it is recommended to
replace this dataset with your own.

Note: This site provides applications using data that has been modified
for use from its original source, www.cityofchicago.org, the official website of
the City of Chicago. The City of Chicago makes no claims as to the content,
accuracy, timeliness, or completeness of any of the data provided at this site.
The data provided at this site is subject to change at any time. It is
understood that the data provided at this site is being used at one's own risk.

You can [read more](
https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips)
about the dataset in [Google BigQuery](https://cloud.google.com/bigquery/).
Explore the full dataset in the
[BigQuery UI](
https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips).


## Testing the pipeline using Stub Executors

### Introduction
**You should complete [template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb) tutorial up to *Step 6* in order to proceed this tutorial.**

This document will provide instructions to test a TensorFlow Extended (TFX) pipeline
using `BaseStubExecuctor`, which generates fake data using the golden test data. This is intended for users to replace executors they don't want to test so that they could save time from running the actual executors. Stub executor is provided with TFX Python package under `tfx.experimental.pipeline_testing.base_stub_executor`.

This tutorial serves as an extension to `template.ipynb` tutorial, thus you will also use [Taxi Trips dataset](
https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago. We strongly encourage you to try modifying the components prior to utilizing stub executors.

### Step 1. Record the pipeline outputs in Google Cloud Storage

Since this tutorial assumes that you have completed `template.ipynb` up to step 6, a successful pipeline run must have been saved in the MLMD, which can be accessed using gRPC server. 

Open a Terminal and run the following commands:

> `$ gcloud container clusters get-credentials <cluster_name> --zone <compute_zone> --project {GOOGLE_CLOUD_PROJECT}`

> `$ nohup kubectl port-forward deployment/metadata-grpc-deployment -n <namespace> {PORT}:8080 &`

Open another terminal and run the following command:
> `$
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=$output_dir \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
`

`--output_dir` should be set to a path in Google Cloud Storage (e.g. `gs://<project_name>-kubeflowpipelines-default/testdata`). 
`--host` and `--port` are hostname and port of the metadata grpc server to connect to MLMD. You may choose any unused port and 'localhost' for hostname.

In `template.ipynb` tutorial, the pipeline name is set as "my_pipeline" by default, so set `pipeline_name="my_pipeline"`. If you have modified the pipeline name when running the template tutorial, you should modify the `--pipeline_name` accordingly.

### Step 2. Enable Stub Executors in Kubeflow DAG Runner
>**Double-click to open `kubeflow_dag_runner.py`**. 
Uncomment `supported_launcher_classes` argument of `KubeflowDagRunnerConfig` to be able to launch stub executors (Tip: search for comments containing `TODO(step 11):`),  Make sure to save `kubeflow_dag_runner.py` after you edit it.



>**Double-click to change directory to `launcher` and double-click again to open `stub_component_launcher.py`.**
Make sure to set `test_data_dir` to the GCS directory where KFP outputs are recorded, or `output_dir` (Tip: search for comments containing `TODO(StubExecutor):`). Customize the list `self.stubbed_component_ids`, or ids of components that should be replaced with BaseStubExecutor(Tip: search for comments containing `TODO(StubExecutor):`). You may additionally insert custom stub executor in `self.stubbed_component_map` with component id as a key and custom stub executor class as value (Tip: search for comments containing `TODO(StubExecutor):`). Make sure to save `stub_component_launcher.py` after you edit it.

### Step 3. Update and run the pipeline with stub executors
Update the existing pipeline with modified pipeline definition with stub executors.
> `
$ tfx pipeline update
--pipeline-path=kubeflow_dag_runner.py
--endpoint={ENDPOINT} --engine=kubeflow
`

Run the following command to create a new execution run of your updated pipeline.

> `
$ tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT} --engine=kubeflow
`
