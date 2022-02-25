# The Tuner TFX Pipeline Component

The Tuner component tunes the hyperparameters for the model.

## Tuner Component and KerasTuner Library

The Tuner component makes extensive use of the Python
[KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) API for
tuning hyperparameters.

Note: The KerasTuner library can be used for hyperparameter tuning regardless of
the modeling API, not just for Keras models only.

## Component

Tuner takes:

*   tf.Examples used for training and eval.
*   A user provided module file (or module fn) that defines the tuning logic,
    including model definition, hyperparameter search space, objective etc.
*   [Protobuf](https://developers.google.com/protocol-buffers) definition of
    train args and eval args.
*   (Optional) [Protobuf](https://developers.google.com/protocol-buffers)
    definition of tuning args.
*   (Optional) transform graph produced by an upstream Transform component.
*   (Optional) A data schema created by a SchemaGen pipeline component and
    optionally altered by the developer.

With the given data, model, and objective, Tuner tunes the hyperparameters and
emits the best result.

## Instructions

A user module function `tuner_fn` with the following signature is required for
Tuner:

```python
...
from keras_tuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

In this function, you define both the model and hyperparameter search spaces,
and choose the objective and algorithm for tuning. The Tuner component takes
this module code as input, tunes the hyperparameters, and emits the best result.

Trainer can take Tuner's output hyperparameters as input and utilize them in
its user module code. The pipeline definition looks like this:

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

You might not want to tune the hyperparameters every time you retrain your
model. Once you have used Tuner to determine a good set of hyperparameters, you
can remove Tuner from your pipeline and use `ImporterNode` to import the Tuner
artifact from a previous training run to feed to Trainer.

```python
hparams_importer = Importer(
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (keras_tuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters,
).with_id('import_hparams')

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## Tuning on Google Cloud Platform (GCP)

When running on the Google Cloud Platform (GCP), the Tuner component can take
advantage of two services:

*   [AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview)
    (via CloudTuner implementation)
*   [AI Platform Training](https://cloud.google.com/ai-platform/training/docs)
    (as a flock manager for distributed tuning)

### AI Platform Vizier as the backend of hyperparameter tuning

[AI Platform Vizier](https://cloud.google.com/ai-platform/optimizer/docs/overview)
is a managed service that performs black box optimization, based on the
[Google Vizier](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/bcb15507f4b52991a0783013df4222240e942381.pdf)
technology.

[CloudTuner](https://github.com/tensorflow/cloud/blob/master/src/python/tensorflow_cloud/tuner/tuner.py)
is an implementation of
[KerasTuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) which talks
to the AI Platform Vizier service as the study backend. Since CloudTuner is a
subclass of `keras_tuner.Tuner`, it can be used as a drop-in replacement in the
`tuner_fn` module, and execute as a part of the TFX Tuner component.

Below is a code snippet which shows how to use `CloudTuner`. Notice that
configuration to `CloudTuner` requires items which are specific to GCP, such as
the `project_id` and `region`.

```python
...
from tensorflow_cloud import CloudTuner

...
def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """An implementation of tuner_fn that instantiates CloudTuner."""

  ...
  tuner = CloudTuner(
      _build_model,
      hyperparameters=...,
      ...
      project_id=...,       # GCP Project ID
      region=...,           # GCP Region where Vizier service is run.
  )

  ...
  return TuneFnResult(
      tuner=tuner,
      fit_kwargs={...}
  )

```

### Parallel tuning on Cloud AI Platform Training distributed worker flock

The KerasTuner framework as the underlying implementation of the Tuner component
has ability to conduct hyperparameter search in parallel. While the stock Tuner
component does not have ability to execute more than one search worker in
parallel, by using the
[Google Cloud AI Platform extension Tuner component](https://github.com/tensorflow/tfx/blob/master/tfx/extensions/google_cloud_ai_platform/tuner/component.py),
it provides the ability to run parallel tuning, using an AI Platform Training
Job as a distributed worker flock manager.
[TuneArgs](https://github.com/tensorflow/tfx/blob/master/tfx/proto/tuner.proto)
is the configuration given to this component. This is a drop-in replacement of
the stock Tuner component.

```python
tuner = google_cloud_ai_platform.Tuner(
    ...   # Same kwargs as the above stock Tuner component.
    tune_args=proto.TuneArgs(num_parallel_trials=3),  # 3-worker parallel
    custom_config={
        # Configures Cloud AI Platform-specific configs . For for details, see
        # https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#traininginput.
        TUNING_ARGS_KEY:
            {
                'project': ...,
                'region': ...,
                # Configuration of machines for each master/worker in the flock.
                'masterConfig': ...,
                'workerConfig': ...,
                ...
            }
    })
...

```

The behavior and the output of the extension Tuner component is the same as the
stock Tuner component, except that multiple hyperparameter searches are executed
in parallel on different worker machines, and as a result, the `num_trials` will
be completed faster. This is particularly effective when the search algorithm is
embarrassingly parallelizable, such as `RandomSearch`. However, if the search
algorithm uses information from results of prior trials, such as Google Vizier
algorithm implemented in the AI Platform Vizier does, an excessively parallel
search would negatively affect the efficacy of the search.

Note: Each trial in each parallel search is conducted on a single machine in the
worker flock, i.e., each trial does not take advantage of multi-worker
distributed training. If multi-worker distribution is desired for each trial,
refer to
[`DistributingCloudTuner`](https://github.com/tensorflow/cloud/blob/b9c8752f5c53f8722dfc0b5c7e05be52e62597a8/src/python/tensorflow_cloud/tuner/tuner.py#L384-L676),
instead of `CloudTuner`.

Note: Both `CloudTuner` and the Google Cloud AI Platform extensions Tuner
component can be used together, in which case it allows distributed parallel
tuning backed by the AI Platform Vizier's hyperparameter search algorithm.
However, in order to do so, the Cloud AI Platform Job must be given access to
the AI Platform Vizier service. See this
[guide](https://cloud.google.com/ai-platform/training/docs/custom-service-account#custom)
to set up a custom service account. After that, you should specify the custom
service account for your training job in the pipeline code. More details see
[E2E CloudTuner on GCP exmaple](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow_gcp.py).

## Links

[E2E Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)

[E2E CloudTuner on GCP Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_kubeflow.py)

[KerasTuner tutorial](https://www.tensorflow.org/tutorials/keras/keras_tuner)

[CloudTuner tutorial](https://github.com/GoogleCloudPlatform/ai-platform-samples/blob/master/notebooks/samples/optimizer/ai_platform_vizier_tuner.ipynb)

[Proposal](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)

More details are available in the
[Tuner API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Tuner).
