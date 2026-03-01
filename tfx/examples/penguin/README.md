# Penguin Classification Example

This directory contains a TFX pipeline configured to train, evaluate and compare
various multi-class classification models trained with TensorFlow 2's Keras,
Jax's Flax and Scikit Learn on the
[Palmer's Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)
dataset.

## Dataset

The
[Palmer's Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)
is a small (344 examples) dataset well suited to demonstrate classification on
tabular datasets. Copies of the original dataset are available in
[Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/penguins) and
[Github](https://github.com/allisonhorst/palmerpenguins).

The dataset contains both categorical and numerical features. The numerical
features are stored as float, and the categorical features and labels are stored
a strings.

**Three examples from the Palmer's Penguins dataset:**

species | island    | bill_length_mm | bill_depth_mm | flipper_length_mm | body_mass_g | sex    | year
------- | --------- | -------------- | ------------- | ----------------- | ----------- | ------ | ----
Adelie  | Torgersen | 39.1           | 18.7          | 181.0             | 3750.0      | male   | 2007
Adelie  | Torgersen | 39.5           | 17.4          | 186.0             | 3800.0      | female | 2007
Adelie  | Torgersen | 40.3           | 18.0          | 195.0             | 3250.0      | female | 2007

This TFX pipeline uses a
[simplified version](https://storage.googleapis.com/download.tensorflow.org/data/palmer_penguins/penguins_processed.csv)
of the dataset. The following preprocessing operations have been applied.

- Remove all the categorical features (`island` and `sex`).
- Converts the label into an integer with the mapping `0=Adelie`, `1=Gentoo`
  and `2=Chinstrap`.
- Remove example with missing values.

Note: While TFX is able to handle natively categorical string features and
missing values, this preprocessing simplifies the TFX pipeline.

## Instruction

NOTE: This examples requires TFX 0.25.0 or later.

Create a Python 3 virtual environment for this example and activate the
`virtualenv`:

<pre class="devsite-terminal devsite-click-to-copy">
virtualenv -p python3.7 penguin
source ./penguin/bin/activate
</pre>

Next, install the dependencies required by the penguin example:

<pre class="devsite-terminal devsite-click-to-copy">
pip install -U tfx[examples]
</pre>

Then, clone the tfx repo and copy penguin/ folder to home directory:

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx ~/tfx-source
cp -r ~/tfx-source/tfx/examples/penguin ~/
</pre>

Execute the pipeline python file and output can be found at `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
python ~/penguin/penguin_pipeline_local.py
</pre>

## Instructions for using Kubeflow V1

NOTE: This example requires TFX 1.0.0 or later.

To compile the pipeline:

<pre class="devsite-terminal devsite-click-to-copy">
python penguin_pipeline_kubeflow.py --use_gcp=True
</pre>

KubeflowDagRunner supports the pipeline with normal components(usually running
on prem) and with Cloud extended components(usually running on GCP).

Validate the output [pipeline-name].tar.gz under the same folder. Upload file to
Kubeflow and run the pipeline.

## Instructions for using Flax

**The Flax support in TFX is experimental.**

The above instructions will use Keras for model definition and training.
Additionally, you can use essentially the same TFX configuration with
[Flax](https://github.com/google/flax) models. The bulk of the differences from
using a Keras model are in `penguin_utils_flax_experimental.py`.

To execute the pipeline with a Flax model:

Install the dependencies required by the flax example:

<pre class="devsite-terminal devsite-click-to-copy">
pip install -U tfx[flax]
</pre>

Then, run the flax pipeline.

<pre class="devsite-terminal devsite-click-to-copy">
python ~/penguin/penguin_pipeline_local.py --model_framework=flax_experimental
</pre>

## Flink Example

This section requires [Apache Flink](https://flink.apache.org/).

All you need is a local Apache Flink cluster with the Flink REST API enabled.
You can download Flink and start a local cluster by running the script:

<pre class="devsite-terminal devsite-click-to-copy">
bash ~/penguin/setup/setup_beam_on_flink.sh
</pre>

The Apache Flink UI can be viewed at http://localhost:8081.

To run tfx e2e on Flink, follow the same instruction above with additional
`runner` flag to execute the pipeline with Flink: `python
~/penguin/penguin_pipeline_local.py --runner=FlinkRunner`

## Spark Example

This section requires [Apache Spark](https://spark.apache.org/).

All you need is a local Apache Spark cluster with the Spark REST API enabled.
You can download Spark and start a local cluster by running the script:

<pre class="devsite-terminal devsite-click-to-copy">
bash ~/penguin/setup/setup_beam_on_spark.sh
</pre>

The Apache Spark UI can be viewed at http://localhost:8081. Check
http://localhost:4040 for the Spark application UI (while a job is running).

To run tfx e2e on Spark, follow the same instruction above with additional
`runner` flag to execute the pipeline with Spark: `python
~/penguin/penguin_pipeline_local.py --runner=SparkRunner`
