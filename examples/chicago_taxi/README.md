# Chicago Taxi Example

The Chicago Taxi example demonstrates the end-to-end workflow and steps of how
to analyze, validate and transform data, train a model, analyze and serve it. It
uses:

* [TensorFlow Data Validation](https://www.tensorflow.org/tfx/data_validation)
  for analyzing and validating data.
* [TensorFlow Transform](https://www.tensorflow.org/tfx/transform) for feature
  preprocessing,
* TensorFlow [Estimators](https://www.tensorflow.org/guide/estimators)
  for training,
* [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/model_analysis) and
  Jupyter for evaluation, and
* [TensorFlow Serving](https://www.tensorflow.org/serving) for serving.

The example shows two modes of deployment:

1. *Local mode* with all necessary dependencies and components deployed locally.
2. *Cloud mode* where all components are deployed on Google Cloud.

## The dataset

This example uses the [Taxi Trips dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago.

Note: This site provides applications using data that has been modified
for use from its original source, www.cityofchicago.org, the official website of
the City of Chicago. The City of Chicago makes no claims as to the content,
accuracy, timeliness, or completeness of any of the data provided at this site.
The data provided at this site is subject to change at any time. It is
understood that the data provided at this site is being used at one’s own risk.

[Read more](https://cloud.google.com/bigquery/public-data/chicago-taxi) about
the dataset in [Google BigQuery](https://cloud.google.com/bigquery/). Explore
the full dataset in the
[BigQuery UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips).

## Local prerequisites

[Apache Beam](https://beam.apache.org/) is used for its distributed processing.
This example requires Python 2.7 since the Apache Beam SDK
([BEAM-1373](https://issues.apache.org/jira/browse/BEAM-1373)) is not yet
available for Python 3.

### Install dependencies

Development for this example will be isolated in a Python virtual environment.
This allows us to experiment with different versions of dependencies.

There are many ways to install `virtualenv`, see the
[TensorFlow install guides](https://www.tensorflow.org/install) for different
platforms, but here are a couple:

* For Linux:

<pre class="devsite-terminal devsite-click-to-copy">
sudo apt-get install python-pip python-virtualenv python-dev build-essential
</pre>

* For Mac:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo easy_install pip</code>
<code class="devsite-terminal">pip install --upgrade virtualenv</code>
</pre>

Create a Python 2.7 virtual environment for this example and activate the
`virtualenv`:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">virtualenv -p python2.7 taxi</code>
<code class="devsite-terminal">source ./taxi/bin/activate</code>
</pre>

Next, install the dependencies required by the Chicago Taxi example:

<pre class="devsite-terminal devsite-click-to-copy">
pip install -r requirements.txt
</pre>

Register the TensorFlow Model Analysis rendering components with Jupyter
Notebook:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis</code>
<code class="devsite-terminal">jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis</code>
</pre>

## Run the local example

The benefit of the local example is that you can edit any part of the pipeline
and experiment very quickly with various components. The example comes with a
small subset of the Taxi Trips dataset as CSV files.

### Analyzing and validating data with TensorFlow Data Validation

`tf.DataValidation` (`tfdv_analyze_and_validate.py`) allows you to analyze and
validate the input dataset. To run this step locally:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./tfdv_analyze_and_validate_local.sh
</pre>

We first compute descriptive [statistics](https://github.com/tensorflow/metadata/tree/master/tensorflow_metadata/proto/v0/statistics.proto)
over the training data. The statistics provide a quick overview of the data in
terms of the features that are present and the shapes of their value
distributions. `tf.DataValidation` uses [Apache Beam](https://beam.apache.org)'s
data-parallel processing framework to scale the computation of statistics over
large datasets. To compute statistics, we use the `GenerateStatistics` Beam
transform which outputs a `DatasetFeatureStatisticsList` protocol buffer.

Next, we infer a [schema](https://github.com/tensorflow/metadata/tree/master/tensorflow_metadata/proto/v0/schema.proto)
for the data using the computed statistics. The schema describes the expected
properties of the data such as which features are expected to be present,
their type, or the expected domains of features, to name a few. In short, the
schema describes the expectations for "correct" data and can thus be used to
detect errors in the data. To infer the schema, we use the `infer_schema`
function which takes as input the `DatasetFeatureStatisticsList` protocol buffer
produced in the above step and outputs a `Schema` protocol buffer.

In general, `tf.DataValidation` uses conservative heuristics to infer stable
data properties from the statistics in order to avoid overfitting the schema to
the specific dataset. It is strongly advised to **review the inferred schema and
refine it as needed** (the schema is stored in the file
`./data/local_tfdv_output/schema.pbtxt`), to
capture any domain knowledge about the data that `tf.DataValidation`'s
heuristics might have missed.

Notice that the `Schema` generated by `tf.DataValidation` is used by the
subsequent steps (`preprocess_local.sh`, `train_local.sh`,
`process_tfma_local.sh`, and `classify_local.sh`) to parse the input examples.

Finally, we check whether the eval dataset conforms to the expectations set in
the schema or whether there exist any data [anomalies](https://github.com/tensorflow/metadata/tree/master/tensorflow_metadata/proto/v0/anomalies.proto).
`tf.DataValidation` performs this check by computing statistics over the eval
dataset and matching the computed statistics against the schema and marking any
discrepancies. To validate the statistics against the schema, we use the
`validate_statistics` function which takes as input the statistics and the
schema, and outputs a `Anomalies` protocol buffer.

`tf.DataValidation` also provides functions to visualize the statistics, schema
and the anomalies in Jupyter. The notebook,
[`chicago_taxi_tfma_local_playground.ipynb`](./chicago_taxi_tfma_local_playground.ipynb)
describes the above steps as well as the visualization functions in detail.

### Preprocessing with TensorFlow Transform

`tf.Transform` (`preprocess.py`) allows preprocessing the results with
full-pass operations over the dataset. To run this step locally:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./preprocess_local.sh
</pre>

Have `tf.Transform` compute the global statistics (mean, standard deviation,
bucket cutoffs, etc.) in the `preprocessing_fn`:

* Take dense float features such as trip distance and time, compute the global
  mean and standard deviations, and scale features to their z scores. This
  allows the SGD optimizer to treat steps in different directions more
  uniformly.
* Create a finite vocabulary for string features. This allow us to treat them as
  categoricals in our model.
* Bucket our longitude and latitude features. This allows the model to fit
  multiple weights to different parts of the lat/long grid.
* Include time-of-day and day-of-week features. This allows the model to more
  easily pick up on seasonality.
* Our label is binary: `1` if the tip is more than 20% of the fare, `0` if it is
  lower.

Preprocessing creates TensorFlow operations to apply the transforms and leaves
TensorFlow placeholders in the graph for the mean, bucket cutoffs, etc.

Call the `Analyze` function in `tf.Transform` to build the MapReduce-style
callbacks to compute the statistics across the entire data set. Apache Beam
allows applications to write such data-transforming code once and handles the
job of placing the code onto workers—whether they are in the cloud, across an
enterprise cluster or on the local machine. In this part of the example, use
`DirectRunner` (see `preprocess_local.sh`) to request that our code runs
on the local machine.  At the end of the Analyze job, the placeholders are
replaced with their respective statistics (mean, standard deviation, etc).

Notice that `tf.Transform` is also used to shuffle the examples
(`beam.transforms.Reshuffle`). This is very important for the efficiency of
non-convex stochastic learners such as SGD with deep neural networks.

Finally, call `WriteTransformFn` to save the transform and
`TransformDataset` to create the examples for training. There are two key
outputs of the preprocessing step:

* The `SavedModel` containing the transform graph;
* Materialized, transformed examples in compressed `TFRecord` files (these are
  inputs to the TensorFlow trainer).

### Train the model

The next step trains the model using TensorFlow. To run this step locally, call:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./train_local.sh
</pre>

The model leverages TensorFlow’s
[Estimators](/guide/estimators) and is created in the
`build_estimator` function in `model.py`. The trainer's input takes the
materialized, transformed examples from the previous step. Notice the pattern of
sharing schema information between preprocessing and training using `taxi.py`
to avoid redundant code.

The `input_fn` function builds a parser that takes `tf.Example` protos from
the materialized `TFRecord` file emitted in our previous preprocessing step. It
also applies the `tf.Transform` operations built in the preprocessing step.
Also, do not forget to remove the label since we don't want the model to treat
it as a feature.

During training, `feature_columns` also come into play and tell the model how
to use features. For example, vocabulary features are fed into the model with
`categorical_column_with_identity` and tell the model to logically treat
this feature as one-hot encoded.

Inside the `eval_input_receiver_fn` callback, emit a TensorFlow graph that
parses raw examples, identifies the features and label, and applies the
`tf.Transform` graph used in the TensorFlow Model Analysis batch job.

Finally, the model emits a graph suitable for serving.

The trainer runs a quick evaluation at the end of the batch job. This limited
evaluation can run only on a single machine. Later in this guide, we will use
TensorFlow Model Analysis—which leverages Apache Beam—for distributed
evaluation.

To recap, the trainer outputs the following:

* A `SavedModel` containing the serving graph to use with
  [TensorFlow Serving](https://www.tensorflow.org/serving).
* A `SavedModel` containing the evaluation graph to use with
  [TensorFlow Model Analysis](https://www.tensorflow.org/tfx/model_analysis).

### TensorFlow Model Analysis evaluation batch job

Now run a batch job to evaluate the model against the entire data set. To run
the batch evaluator:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./process_tfma_local.sh
</pre>

This step is executed locally with a small CSV dataset. TensorFlow Model
Analysis runs a full pass over the dataset and computes the metrics. In the
Cloud section below, this step is run as a distributed job over a large data
set.

Like TensorFlow Transform, `tf.ModelAnalysis` uses Apache Beam to run the
distributed computation. The evaluation uses raw examples as input and *not*
transformed examples. The input is a local CSV file and uses the `SavedModel`
from the previous step, applies the `tf.Transform` graph, then runs the model.

It *is* possible to analyze models in terms of transformed features instead of
raw features, but that process is not described in this guide.

In `process_tfma.py` we specified a `slice_spec` that tells the
`tf.ModelAnalysis` job the slices to visualize. Slices are subsets of the
data based on feature values. `tf.ModelAnalysis` computes metrics for each of
those slices.

This job outputs a file that can be visualized in Jupyter in the
`tf.model_analysis` renderer in the next stage.

### TensorFlow Model Analysis rendered metrics

Run the Jupyter notebook locally to view the sliced results:

<pre class="devsite-terminal devsite-click-to-copy">
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
</pre>

This prints a URL like `http://0.0.0.0:8888/?token=...`. You may need to
enter a username and password the first time you visit the page (these prompts
accept anything, so you can use "test" for both).

From the *File* tab, open the `chicago_taxi_tfma.ipynb` notebook file.
Follow the instructions in the notebook until you visualize the slicing metrics
browser with `tfma.view.render_slicing_metrics` and the time series graph
with `tfma.view.render_time_series`.

Note: This notebook is self-contained and does not rely on running the prior
scripts.

### Serve the TensorFlow model

Now serve the created model with
[TensorFlow Serving](https://www.tensorflow.org/serving). For this example, run
the server in a Docker container that is run locally. Instructions for
installing Docker locally are found in the
[Docker install documentation](https://docs.docker.com/install).

In the terminal, run the following script to start a server:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./start_model_server_local.sh
</pre>

This script pulls a TensorFlow Serving serving image and listens for for gRPC
requests on `localhost` port 9000. The model server loads the model exported
from the trainer.

To send a request to the server for model inference, run:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./classify_local.sh
</pre>

For more information, see [TensorFlow Serving](/serving/).

### Playground notebook

The notebook,
[`chicago_taxi_tfma_local_playground.ipynb`](./chicago_taxi_tfma_local_playground.ipynb),
is available which calls the same scripts above. It has a more detailed
description of the APIs—like custom metrics and plots—and the UI components.

Note: This notebook is self-contained and does not rely on running the prior
scripts.


## Cloud prerequisites

This section requires the [local prerequisites](#local_prerequisites) and adds a
few more for the [Gooogle Cloud Platform](https://cloud.google.com/).

Follow the Google Cloud Machine Learning Engine
[setup guide](https://cloud.google.com/ml-engine/docs/how-tos/getting-set-up)
and the Google Cloud Dataflow
[setup guide](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)
before trying the example.

The example's execution speed may be limited by default
[Google Compute Engine](https://cloud.google.com/compute) quota. We recommend
setting a sufficient quota for approximately 250 Dataflow VMs: *250 CPUs, 250 IP
Addresses, and 62500 GB of Persistent Disk*. For more details, please see the
[GCE Quota](https://cloud.google.com/compute/quotas) and
[Dataflow Quota](https://cloud.google.com/dataflow/quotas) documentation.

This example uses [Google Cloud Storage](https://cloud.google.com/storage/)
Buckets to store data, and uses local environment variables to pass paths from
job to job.

Authenticate and switch to your project:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">gcloud auth application-default login</code>
<code class="devsite-terminal">gcloud config set project $PROJECT_NAME</code>
</pre>

Create the `gs://` bucket:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">export MYBUCKET=gs://$(gcloud config list --format 'value(core.project)')-chicago-taxi</code>
<code class="devsite-terminal">gsutil mb $MYBUCKET</code>
</pre>

Activate the `virtualenv` (created above) to setup the shell environment:

<pre class="devsite-terminal devsite-click-to-copy">
source ./bin/taxi/activate
</pre>

## Run the Cloud example

This Taxi Trips example is run in the cloud. Unlike the local example, the input
is a much larger dataset hosted on Google BigQuery.

### Analyzing and validating data with TensorFlow Data Validation on Google Cloud Dataflow

Use the same code from the local `tf.DataValidation`
(in `tfdv_analyze_and_validate.py`) to do the distributed analysis and
validation. To start the job, run:

<pre class="devsite-terminal devsite-click-to-copy">
source ./tfdv_analyze_and_validate_dataflow.sh
</pre>

You can see the status of the running job on the
[Google Cloud Console](https://console.cloud.google.com/dataflow).

In this case, the data is read from Google BigQuery instead of a small, local
CSV file. Also, unlike our local example above, we are using the
`DataflowRunner` to start distributed processing over several workers in the
cloud.

The outputs are the same as the local job, but stored in Google Cloud Storage:

* Training data statistics:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $TFDV_OUTPUT_PATH/train_stats.tfrecord
</pre>

* Schema:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $TFDV_OUTPUT_PATH/schema.pbtxt
</pre>

* Eval data statistics:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $TFDV_OUTPUT_PATH/eval_stats.tfrecord
</pre>

* Anomalies:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $TFDV_OUTPUT_PATH/anomalies.pbtxt
</pre>

Notice that the `Schema` generated by `tf.DataValidation` is used by the
subsequent steps (`preprocess_dataflow.sh`, `train_mlengine.sh`,
`process_tfma_dataflow.sh`, and `classify_mlengine.sh`) to parse the input
examples.

### Preprocessing with TensorFlow Transform on Google Cloud Dataflow

Use the same code from the local `tf.Transform` (in `preprocess.py`) to do
the distributed transform. To start the job, run:

<pre class="devsite-terminal devsite-click-to-copy">
source ./preprocess_dataflow.sh
</pre>

You can see the status of the running job on the
[Google Cloud Console](https://console.cloud.google.com/dataflow).

Unlike our local example above, we are using the `DataflowRunner`
to start distributed processing over several workers in the cloud.

The outputs are the same as the local job, but stored in Google Cloud Storage:

* `SavedModel` containing the transform graph:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $TFT_OUTPUT_PATH/transform_fn
</pre>

* Materialized, transformed examples (train_transformed-\*):

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $TFT_OUTPUT_PATH
</pre>

### Model training on the Google Cloud Machine Learning Engine

Run the distributed TensorFlow trainer in the cloud:

<pre class="devsite-terminal devsite-click-to-copy">
source ./train_mlengine.sh
</pre>

See the status of the running job in the
[Google Cloud Console](https://console.cloud.google.com/mlengine).

The trainer is running in the cloud using ML Engine and *not* Dataflow for the
distributed computation.

Again, our outputs are identical to the local run:

* `SavedModel` containing the serving graph for TensorFlow Serving:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $WORKING_DIR/serving_model_dir/export/chicago-taxi
</pre>

* `SavedModel` containing the evaluation graph for TensorFlow Model Analysis:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls $WORKING_DIR/eval_model_dir
</pre>

### Model Evaluation with TensorFlow Model Analysis on Google Cloud Dataflow

Run a distributed batch job to compute sliced metrics across the large Google
BigQuery dataset. In this step, `tf.ModelAnalysis` takes advantage of the
`DataflowRunner` to control its workers.

<pre class="devsite-terminal devsite-click-to-copy">
source ./process_tfma_dataflow.sh
</pre>

The output is the `eval_result` file rendered by the notebook in the next
step:

<pre class="devsite-terminal devsite-click-to-copy">
gsutil ls -l $TFT_OUTPUT_PATH/eval_result_dir
</pre>

### Render TensorFlow Model Analysis results in a local Jupyter notebook

The steps for looking at your results in a notebook are identical to the ones
from running the local job. Go to the
[`chicago_taxi_tfma.ipynb`](./chicago_taxi_tfma.ipynb)
notebook and set up the output directory to see the results.

### Model serving on the Google Cloud Machine Learning Engine

To serve the model from the cloud, run:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./start_model_server_mlengine.sh
</pre>

To send a request to the cloud:

<pre class="devsite-terminal devsite-click-to-copy">
bash ./classify_mlengine.sh
</pre>
