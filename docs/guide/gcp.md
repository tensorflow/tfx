# Running a TFX Pipeline with Google Cloud Platform

## Install

You can also configure TFX to run the components on Google Cloud Platform (GCP).  The GCP products used are Dataflow and Cloud ML Engine.  For this example, we will continue to use Airflow as the workflow engine.  This example builds on top of the prior “Running locally” example, and you should understand how that pipeline runs before moving to the GCP version.

### Google Cloud Platform

In addition, you will need to install the Google APIs client libraries for Python.  Instructions are [here](https://developers.google.com/api-client-library/python/start/installation).

Additionally, you will need to enable the Dataflow API if you are working with a new GCP project.  On [https://console.cloud.google.com](https://console.cloud.google.com), the Dataflow API can be found in the API library.  Enable it.

### Authorizing the pipeline to your project

You must authorize your local machine to access your project on GCP.  See the
[Getting Started](https://cloud.google.com/docs/authentication/getting-started) guide for information on how to do this.  You’ll also need to ensure that your GOOGLE_APPLICATION_CREDENTIALS have been set for your environment.

## Configuring your pipeline

Running the TFX pipeline on GCP is very similar to running locally.  All you need to do is configure your pipeline to use the GCP equivalents of the TFX components used in the local pipeline.

### Configuring the pipeline to use Google Cloud Platform

Most likely, your data already exists in a Google Cloud Storage bucket.  The major configuration change moving from local to GCP is just to update where your input data and module function is located, and where you want your pipeline output to be stored.

```python
# Google Cloud Storage buckets
input_bucket = 'gs://put-your-bucket-here'
output_bucket = 'gs://put-your-bucket-here'
pipeline_root = os.path.join(output_bucket, 'tfx/pipelines')
tmp_bucket = os.path.join(output_bucket, 'tmp')

# Other GCP parameters
region = 'us-central1'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
taxi_module_file = os.path.join(input_bucket, 'taxi_utils.py')
```

Note: this example requires your module file is also stored on Google Cloud Storage.  Also, we recommend creating a region variable to ensure the TFX components are all run in the same region.

### Configuring your pipeline to use Google Dataflow

To enable Dataflow, we need to override the beam’s local directrunner with DataflowRunner.  To do this, add `beam_pipeline_args` as an additional argument in your pipeline.  You can then configure your Dataflow pipeline using the standard
[Dataflow pipeline execution parameters](
https://cloud.google.com/dataflow/docs/guides/specifying-exec-params#configuring-pipelineoptions-for-execution-on-the-cloud-dataflow-service).  

```python
@PipelineDecorator(
    pipeline_name='chicago_taxi_gcp',
    enable_cache=True,
    metadata_db_root=metadata_db_root,
    pipeline_root=pipeline_root,
    additional_pipeline_args={
        'beam_pipeline_args': [
            '--runner=DataflowRunner', 
            '--setup=' + 'your-setup-file', 
            '--experiments=shuffle_mode=auto',
            '--project=' + 'your-project-id',
            '--temp_location=' + tmp_bucket,
            '--staging_location=' + tmp_bucket,
            '--region=' + region
        ],
    })
```

### Configuring your pipeline to use Google Cloud ML Engine

To enable Cloud ML Engine, we need to make two changes.  First, we need to reconfigure the pipeline to use components that will submit jobs to Cloud ML Engine:

from:

```python
from tfx.components.pusher.component import Pusher
from tfx.components.trainer.component import Trainer
```

to:

```python
from tfx.extensions.google_cloud_ml_engine.pusher.component import Pusher
from tfx.extensions.google_cloud_ml_engine.trainer.component import Trainer
```

Also, we need to specify Cloud ML Engine job parameters for both of these components.  First, create two dictionaries that contain the job parameters.  All CMLE training and serving arguments can be passed through to 

```python
cmle_training_args = {
    'region': region,
    'jobDir': os.path.join(output_bucket, 'tmp'),
    'runtimeVersion': '1.12',
    'pythonVersion': '2.7',
    'project': 'your-project-id',
    'pythonModule': None,  # Reserved; will be populated by TFX
    'args': None,  # Reserved; will be populated by TFX
}

cmle_serving_args = {
    'model_name': 'chicago_taxi',
    'project_id': 'your-project-id',
    'runtime_version': '1.12',
}
```

Then add them into the new components as a custom config parameter:

```python
  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=taxi_module_file,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_steps=10000,
      eval_steps=5000,
      custom_config={'cmle_training_args': cmle_training_args})

...

  # Checks whether the model passed the validation steps and pushes the model
  # to Google Cloud ML Engine if check passed.
  pusher = Pusher(
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      serving_model_dir=serving_model_dir,
      custom_config={'cmle_serving_args': cmle_serving_args})
```

Now when you run your pipeline, all computation will be executed on Google Cloud Platform.  The examples only use 15,000 records, which is 0.01% of the entire dataset.  Using GCP, you can process the full dataset with the same pipeline.