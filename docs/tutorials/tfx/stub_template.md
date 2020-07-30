
## Testing the pipeline using Stub Executors

### Introduction
**You should complete [template.ipynb](https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb) tutorial up to *6* in order to proceed this tutorial.**

This document will provide instructions to test a TensorFlow Extended (TFX) pipeline
using `BaseStubExecuctor`, which generates fake data using the golden test data. This is intended for users to replace executors they don't want to test so that they could save time from running the actual executors. Stub executor is provided with TFX Python package under `tfx.experimental.pipeline_testing.base_stub_executor`.

This tutorial serves as an extension to `template.ipynb` tutorial, thus you will also use [Taxi Trips dataset](
https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago. We strongly encourage you to try modifying the components prior to utilizing stub executors.

### 1. Record the pipeline outputs in Google Cloud Storage

Since this tutorial assumes that you have completed `template.ipynb` up to 6, a successful pipeline run must have been saved in the MLMD, which can be accessed using gRPC server. 

Open a Terminal and run the following commands:

```bash
gcloud container clusters get-credentials <cluster_name> --zone <compute_zone> --project {GOOGLE_CLOUD_PROJECT}
```

```bash
nohup kubectl port-forward deployment/metadata-grpc-deployment -n <namespace> {PORT}:8080 &
```

Then, you can either:

1.  Open another terminal and clone the tfx GitHub repository.
 Inside the directory, run the following command:
```bash
python tfx/experimental/pipeline_testing/pipeline_recorder.py \
--output_dir=gs://<project_name>-kubeflowpipelines-default/testdata \
--host=$host \
--port=$port \
--pipeline_name=$pipeline_name
```

2. Run the python file
```python
from tfx.experimental.pipeline_testing import pipeline_recorder_utils
    pipeline_recorder_utils.record_pipeline(
        output_dir=gs://<project_name>-kubeflowpipelines-default/testdata,
        metadata_db_uri=None,
        host=$host,
        port=$port,
        pipeline_name=$pipeline_name,
        run_id=None)
```


`$output_dir` should be set to a path in Google Cloud Storage where the pipeline outputs are to be recorded, so make sure to replace `<project_name>` with Google Cloud project name.

`$host` and `$port` are hostname and port of the metadata grpc server to connect to MLMD. You may choose any unused port and "localhost" for hostname.

In `template.ipynb` tutorial, the pipeline name is set as "my_pipeline" by default, so set `pipeline_name="my_pipeline"`. If you have modified the pipeline name when running the template tutorial, you should modify the `--pipeline_name` accordingly.

### 2. Enable Stub Executors in Kubeflow DAG Runner

First, make sure that the predefined template has been copied to your project directory using `tfx template copy` CLI command. It is necessary to edit the following two files in the copied source files.

1.  In `kubeflow_dag_runner.py`, add `StubComponentLauncher` class to `KubeflowDagRunnerConfig`'s `supported_launcher_class` to enable launch of stub executors:

``` python
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
      # TODO(StubExecutor): Uncomment below to use stub executors.
      supported_launcher_classes=[
        stub_component_launcher.StubComponentLauncher
      ],
```

2.  In `launcher/stub_component_launcher.py`, modify the default list of component ids that are to be replaced with BaseStubExecutor. If you want to replace component executor with a custom stub executor, which inherits BaseStubExecutor, modify stubbed_component_map to include component id to custom stub executor mapping. If 
```python
# GCS directory where KFP outputs are recorded
test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
# TODO(StubExecutor): customize self.stubbed_component_ids to replace components
# with BaseStubExecutor
stubbed_component_ids = ['CsvExampleGen', 'StatisticsGen',
                         'SchemaGen', 'ExampleValidator',
                         'Trainer', 'Transform', 'Evaluator', 'Pusher']
# TODO(StubExecutor): (Optional) Use stubbed_component_map to insert custom stub
# executor class as a value and component id as a key.
stubbed_component_map = {}
```

#### In Jupyter lab file editor:
>**Double-click to open `kubeflow_dag_runner.py`**. 
Uncomment `supported_launcher_classes` argument of `KubeflowDagRunnerConfig` to be able to launch stub executors (Tip: search for comments containing `TODO(11):`),  Make sure to save `kubeflow_dag_runner.py` after you edit it.



>**Double-click to change directory to `launcher` and double-click again to open `stub_component_launcher.py`.**
Make sure to set `test_data_dir` to the GCS directory where KFP outputs are recorded, or `output_dir` (Tip: search for comments containing `TODO(StubExecutor):`). Customize the list `self.stubbed_component_ids`, or ids of components that should be replaced with BaseStubExecutor(Tip: search for comments containing `TODO(StubExecutor):`). You may additionally insert custom stub executor in `self.stubbed_component_map` with component id as a key and custom stub executor class as value (Tip: search for comments containing `TODO(StubExecutor):`). Make sure to save `stub_component_launcher.py` after you edit it.

### 3. Update and run the pipeline with stub executors
Update the existing pipeline with modified pipeline definition with stub executors.
```bash
tfx pipeline update
--pipeline-path=kubeflow_dag_runner.py
--endpoint={ENDPOINT} --engine=kubeflow
```

Run the following command to create a new execution run of your updated pipeline.

```bash
tfx run create --pipeline-name {PIPELINE_NAME} --endpoint={ENDPOINT} --engine=kubeflow
```
