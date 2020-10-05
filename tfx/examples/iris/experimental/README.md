# Iris flowers classification Example

**Warning: This code is experimental and is not ready for use. We welcome
feedback on this example.**

The Iris flowers classification example introduces the TFX programming
environment and shows you how to solve the classification problem in
TensorFlow TFX.

The Iris flowers classification example demonstrates the end-to-end workflow
and steps of how to classify Iris flower subspecies.

## Instructions

Clone the tfx repo to the home directory and copy the iris directory from the
tfx source to the home directory.

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx ~/tfx-source && pushd ~/tfx-source
cp -r ~/tfx-source/tfx/examples/iris ~/
</pre>

Next, create a Python 3 virtual environment for this example and activate the
`virtualenv`:

<pre class="devsite-terminal devsite-click-to-copy">
virtualenv -p python3.7 iris
source ./iris/bin/activate
</pre>

Then, install the dependencies required by the Iris example:

<pre class="devsite-terminal devsite-click-to-copy">
pip install tfx==0.24.0 \
  numpy==1.16.5 \
  scikit-learn==0.20.4
</pre>

### Local Example
Execute the pipeline python file. Output can be found at `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
python ~/iris/experimental/iris_pipeline_sklearn_local.py
</pre>

### GCP Example
This example uses a custom container image instead of the default TFX ones found
[here](gcr.io/tfx-oss-public/tfx). This custom container ensures the proper
version of scikit-learn is installed. Run the following commands to build this
image and upload it to Google Container Registry (GCR).

<pre class="devsite-terminal devsite-click-to-copy">
gcloud auth configure-docker
docker build \
  --tag tfx-example-sklearn \
  --build-arg TFX_VERSION=`python -c 'import tfx; print(tfx.__version__)'` \
  .
docker tag tfx-example-sklearn gcr.io/[PROJECT-ID]/tfx-example-sklearn
docker push gcr.io/[PROJECT-ID]/tfx-example-sklearn
</pre>

Note that the custom container extends an official TFX container image based on
the local TFX version. If an unreleased version of TFX is being used
(e.g. installing from HEAD), `Dockerfile` may need to be modified to install the
unreleased version.

Set the project id and bucket in `iris_pipeline_sklearn_gcp.py`. Then, run the
following commands to copy the `~/iris` directory to GCS and execute the
pipeline python file. Output can be found at `[BUCKET]/tfx` and metadata will
be stored in `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
vi ~/iris/experimental/iris_pipeline_sklearn_gcp.py
gsutil cp -r ~/iris/data gs://[BUCKET]/iris
gsutil cp -r ~/iris/experimental gs://[BUCKET]/iris
python ~/iris/experimental/iris_pipeline_sklearn_gcp.py
</pre>

Note that `gsutil cp -r ~/iris/experimental gs://[BUCKET]/iris` will need to be
run every time updates are made to `iris_pipeline_sklearn_gcp.py`.
