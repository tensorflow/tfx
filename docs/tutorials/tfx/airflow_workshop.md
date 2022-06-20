# TFX Airflow Tutorial

[![Python](https://img.shields.io/pypi/pyversions/tfx.svg?style=plastic)](
https://github.com/tensorflow/tfx)
[![PyPI](https://badge.fury.io/py/tfx.svg)](https://badge.fury.io/py/tfx)

## Introduction

This tutorial is designed to introduce TensorFlow Extended (TFX)
and help you learn to create your own machine learning
pipelines.  It runs locally, and shows integration with TFX and TensorBoard
as well as interaction with TFX in Jupyter notebooks.

Key Term: A TFX pipeline is a Directed Acyclic Graph, or "DAG".  We will often
refer to pipelines as DAGs.

You'll follow a typical ML development process,
starting by examining the dataset, and end up with a complete
working pipeline.  Along the way you'll explore ways to debug
and update your pipeline, and measure performance.

### Learn more

Please see the [TFX User Guide](https://www.tensorflow.org/tfx/guide) to learn
more.

## Step by step

You'll gradually create your pipeline by working step by step,
following a typical ML development process.  Here are the steps:

1. [Setup your environment](#step_1_setup_your_environment)
1. [Bring up initial pipeline skeleton](
#step_2_bring_up_initial_pipeline_skeleton)
1. [Dive into your data](#step_3_dive_into_your_data)
1. [Feature engineering](#step_4_feature_engineering)
1. [Training](#step_5_training)
1. [Analyzing model performance](#step_6_analyzing_model_performance)
1. [Ready for production](#step_7_ready_for_production)

## Prerequisites

* Linux / MacOS
* Virtualenv
* Python 3.5+
* Git

### Required packages

Depending on your environment you may need to install several packages:

```bash
sudo apt-get install \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev \
    python3-pip git software-properties-common
```

If you are running Python 3.6 you should install python3.6-dev:

```bash
sudo apt-get install python3.6-dev
```

If you are running Python 3.7 you should install python3.7-dev:

```bash
sudo apt-get install python3.7-dev
```

In addition, if your system has a GCC version < 7 you should update GCC.
Otherwise you will see errors when running `airflow webserver`. You can check
your current version with:

```bash
gcc --version
```

If you need to update GCC, you can run this:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7
sudo apt install g++-7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
```

### MacOS Environment

If you do not have Python 3 and git already installed, you can install them
using the [Homebrew](https://brew.sh/) package manager:

```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python
brew install git
```

MacOS sometimes has problems forking threads when running Airflow, depending on
the configuration. To avoid those problems you should edit your
`~/.bash_profile` and add the following line to the end of the file:

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
```

## Tutorial materials

The code for this tutorial is available at:
[https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop](
https://github.com/tensorflow/tfx/tree/master/tfx/examples/airflow_workshop)

The code is organized by the steps that you're working on, so
for each step you'll have the code you need and instructions
on what to do with it.

The tutorial files include both an exercise and the solution to
the exercise, in case you get stuck.

#### Exercise

* taxi_pipeline.py
* taxi_utils.py
* taxi DAG

#### Solution

* taxi_pipeline_solution.py
* taxi_utils_solution.py
* taxi_solution DAG

## What you're doing

You’re learning how to create an ML pipeline using TFX

* TFX pipelines are appropriate when you will be deploying a production ML
application
* TFX pipelines are appropriate when datasets are large
* TFX pipelines are appropriate when training/serving consistency is important
* TFX pipelines are appropriate when version management for inference is
important
* Google uses TFX pipelines for production ML

You’re following a typical ML development process

* Ingesting, understanding, and cleaning our data
* Feature engineering
* Training
* Analyze model performance
* Lather, rinse, repeat
* Ready for production

### Adding the code for each step

The tutorial is designed so that all the code is included in the files, but all
the code for steps 3-7 is commented out and marked with inline comments. The
inline comments identify which step the line of code applies to. For example,
the code for step 3 is marked with the comment `# Step 3`.

The code that you will add for each step typically falls into 3 regions of the
code:

* imports
* The DAG configuration
* The list returned from the create_pipeline() call
* The supporting code in taxi_utils.py

As you go through the tutorial you'll uncomment the lines of code that apply to
the tutorial step that you're currently working on.  That will add the code for
that step, and update your pipeline.  As you do that **we strongly encourage
you to review the code that you're uncommenting**.

## Chicago Taxi Dataset
<!-- Image free for commercial use, does not require attribution:
https://pixabay.com/photos/new-york-cab-cabs-taxi-urban-city-2087998/ -->
![Taxi](images/airflow_workshop/taxi.jpg)
![Chicago taxi](images/airflow_workshop/chicago.png)

You're using the [Taxi Trips dataset](
https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago.

Note: This site provides applications using data that has been modified
for use from its original source, www.cityofchicago.org, the official website of
the City of Chicago. The City of Chicago makes no claims as to the content,
accuracy, timeliness, or completeness of any of the data provided at this site.
The data provided at this site is subject to change at any time. It is
understood that the data provided at this site is being used at one’s own risk.

You can [read more](https://cloud.google.com/bigquery/public-data/chicago-taxi)
about the dataset in [Google BigQuery](https://cloud.google.com/bigquery/).
Explore the full dataset in the [BigQuery UI](
https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips).

### Model Goal - Binary classification

Will the customer tip more or less than 20%?

## Step 1: Setup your environment

The setup script (`setup_demo.sh`) installs TFX and
[Airflow](https://airflow.apache.org/), and configures
Airflow in a way that makes it easy to work with for this tutorial.

In a shell:

```bash
cd
virtualenv -p python3 tfx-env
source ~/tfx-env/bin/activate

git clone https://github.com/tensorflow/tfx.git
cd ~/tfx
# These instructions are specific to the 0.21 release
git checkout -f origin/r0.21
cd ~/tfx/tfx/examples/airflow_workshop/setup
./setup_demo.sh
```

You should review `setup_demo.sh` to see what it's doing.

## Step 2: Bring up initial pipeline skeleton

### Hello World

In a shell:

```bash
# Open a new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow webserver -p 8080

# Open another new terminal window, and in that window ...
source ~/tfx-env/bin/activate
airflow scheduler

# Open yet another new terminal window, and in that window ...
# Assuming that you've cloned the TFX repo into ~/tfx
source ~/tfx-env/bin/activate
cd ~/tfx/tfx/examples/airflow_workshop/notebooks
jupyter notebook
```

You started Jupyter notebook in this step.  Later you will be running the
notebooks in this folder.

### In a browser:

* Open a browser and go to http://127.0.0.1:8080

#### Troubleshooting

If you have any issues with loading the Airflow console in your web browser,
or if there were any errors when you ran `airflow webserver`, then you may have
another application running on port 8080.  That's the default port for Airflow,
but you can change it to any other user port that's not being used.  For
example, to run Airflow on port 7070 you could run:

```bash
airflow webserver -p 7070
```

#### DAG view buttons

![DAG buttons](images/airflow_workshop/airflow_dag_buttons.png)

* Use the button on the left to _enable_ the DAG
* Use the button on the right to _refresh_ the DAG when you make changes
* Use the button on the right to _trigger_ the DAG
* Click on taxi to go to the graph view of the DAG

![Graph refresh button](images/airflow_workshop/graph_refresh_button.png)

#### Airflow CLI

You can also use the [Airflow CLI](https://airflow.apache.org/cli.html) to
enable and trigger your DAGs:

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### Waiting for the pipeline to complete

After you've triggered your pipeline in the DAGs view, you can watch as your
pipeline completes processing.  As each component runs the outline color of the
component in the DAG graph will change to show its state.  When a component has
finished processing the outline will turn dark green to show that it's done.

Note: You need to use the _graph refresh_ button on the right or refresh the
page to see the updated status of the components as they run.

So far you only have the CsvExampleGen component in our pipeline, so you need
to wait for it to turn dark green (~1 minutes).

![Setup complete](images/airflow_workshop/step2.png)

## Step 3: Dive into your data

The first task in any data science or ML project is to understand
and clean the data.

* Understand the data types for each feature
* Look for anomalies and missing values
* Understand the distributions for each feature

### Components

![Data Components](images/airflow_workshop/examplegen1.png)
![Data Components](images/airflow_workshop/examplegen2.png)

* [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen)
ingests and splits the input dataset.
* [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen)
calculates statistics for the dataset.
* [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen)
SchemaGen examines the statistics and creates a data schema.
* [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval)
looks for anomalies and missing values in the dataset.

### In an editor:

* In ~/airflow/dags uncomment the lines marked `Step 3` in
`taxi_pipeline.py`
* Take a moment to review the code that you uncommented

### In a browser:

* Return to DAGs list page in Airflow by clicking on "DAGs" link in the top
left corner
* Click the refresh button on the right side for the taxi DAG
  * You should see "DAG [taxi] is now fresh as a daisy"
* Trigger taxi
* Wait for pipeline to complete
  * All dark green
  * Use refresh on right side or refresh page

![Dive into data](images/airflow_workshop/step3.png)

### Back on Jupyter:

Earlier, you ran `jupyter notebook`, which opened a Jupyter session in a browser
tab.  Now return to that tab in your browser.

* Open step3.ipynb
* Follow the notebook

![Dive into data](images/airflow_workshop/step3notebook.png)

### More advanced example

The example presented here is really only meant to get you started. For a more
advanced example see the [TensorFlow Data Validation Colab](
https://www.tensorflow.org/tfx/tutorials/data_validation/chicago_taxi).

For more information on using TFDV to explore and validate a
dataset, [see the examples on tensorflow.org](
https://www.tensorflow.org/tfx/data_validation).

## Step 4: Feature engineering

You can increase the predictive quality of your data and/or reduce
dimensionality with feature engineering.

* Feature crosses
* Vocabularies
* Embeddings
* PCA
* Categorical encoding

One of the benefits of using TFX is that you will write your transformation
code once, and the resulting transforms will be consistent between training
and serving.

### Components

![Transform](images/airflow_workshop/transform.png)

* [Transform](https://www.tensorflow.org/tfx/guide/transform)
performs feature engineering on the dataset.

### In an editor:

* In ~/airflow/dags uncomment the lines marked `Step 4` in both
`taxi_pipeline.py` and `taxi_utils.py`
* Take a moment to review the code that you uncommented

### In a browser:

* Return to DAGs list page in Airflow
* Click the refresh button on the right side for the taxi DAG
  * You should see "DAG [taxi] is now fresh as a daisy"
* Trigger taxi
* Wait for pipeline to complete
  * All dark green
  * Use refresh on right side or refresh page

![Feature Engineering](images/airflow_workshop/step4.png)

### Back on Jupyter:

Return to the Jupyter tab in your browser.

* Open step4.ipynb
* Follow the notebook

### More advanced example

The example presented here is really only meant to get you started. For a more
advanced example see the [TensorFlow Transform Colab](
https://www.tensorflow.org/tfx/tutorials/transform/census).

## Step 5: Training

Train a TensorFlow model with your nice, clean, transformed data.

* Include the transformations from step 4 so that they are applied consistently
* Save the results as a SavedModel for production
* Visualize and explore the training process using TensorBoard
* Also save an EvalSavedModel for analysis of model performance

### Components

*   [Trainer](https://www.tensorflow.org/tfx/guide/trainer) trains the model
    using TensorFlow
    [Estimators](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/estimators.md)

### In an editor:

* In ~/airflow/dags uncomment the lines marked `Step 5` in both 
`taxi_pipeline.py` and `taxi_utils.py`
* Take a moment to review the code that you uncommented

### In a browser:

* Return to DAGs list page in Airflow
* Click the refresh button on the right side for the taxi DAG
  * You should see "DAG [taxi] is now fresh as a daisy"
* Trigger taxi
* Wait for pipeline to complete
  * All dark green
  * Use refresh on right side or refresh page

![Training a Model](images/airflow_workshop/step5.png)

### Back on Jupyter:

Return to the Jupyter tab in your browser.

* Open step5.ipynb
* Follow the notebook

![Training a Model](images/airflow_workshop/step5tboard.png)

### More advanced example

The example presented here is really only meant to get you started. For a more
advanced example see the
[TensorBoard Tutorial](https://www.tensorflow.org/tensorboard/get_started).

## Step 6: Analyzing model performance

Understanding more than just the top level metrics.

* Users experience model performance for their queries only
* Poor performance on slices of data can be hidden by top level metrics
* Model fairness is important
* Often key subsets of users or data are very important, and may be small
    * Performance in critical but unusual conditions
    * Performance for key audiences such as influencers
* If you’re replacing a model that is currently in production, first make sure
that the new one is better
* Evaluator tells the Pusher component if the model is OK

### Components

* [Evaluator](https://www.tensorflow.org/tfx/guide/evaluator)
performs deep analysis of the training results, and ensures that the model is
"good enough" to be pushed to production.

### In an editor:

* In ~/airflow/dags uncomment the lines marked `Step 6` in both
`taxi_pipeline.py`
* Take a moment to review the code that you uncommented

### In a browser:

* Return to DAGs list page in Airflow
* Click the refresh button on the right side for the taxi DAG
  * You should see "DAG [taxi] is now fresh as a daisy"
* Trigger taxi
* Wait for pipeline to complete
  * All dark green
  * Use refresh on right side or refresh page

![Analyzing model performance](images/airflow_workshop/step6.png)

### Back on Jupyter:

Return to the Jupyter tab in your browser.

* Open step6.ipynb
* Follow the notebook

![Analyzing model performance](images/airflow_workshop/step6notebook.png)

### More advanced example

The example presented here is really only meant to get you started.
For a more advanced example see the [TFMA Chicago Taxi Tutorial](
https://www.tensorflow.org/tfx/tutorials/model_analysis/chicago_taxi).

## Step 7: Ready for production

If the new model is ready, make it so.

* Pusher deploys SavedModels to well-known locations

Deployment targets receive new models from well-known locations

* TensorFlow Serving
* TensorFlow Lite
* TensorFlow JS
* TensorFlow Hub

### Components

* [Pusher](https://www.tensorflow.org/tfx/guide/pusher)
deploys the model to a serving infrastructure.

### In an editor:

* In ~/airflow/dags uncomment the lines marked `Step 7` in both
`taxi_pipeline.py`
* Take a moment to review the code that you uncommented

### In a browser:

* Return to DAGs list page in Airflow
* Click the refresh button on the right side for the taxi DAG
  * You should see "DAG [taxi] is now fresh as a daisy"
* Trigger taxi
* Wait for pipeline to complete
  * All dark green
  * Use refresh on right side or refresh page

![Ready for production](images/airflow_workshop/step7.png)

## Next Steps

You have now trained and validated your model, and exported a `SavedModel` file
under the `~/airflow/saved_models/taxi` directory.  Your model is now
ready for production.  You can now deploy your model to any of the TensorFlow
deployment targets, including:

* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), for
serving your model on a server or server farm and processing REST and/or gRPC
inference requests.
* [TensorFlow Lite](https://www.tensorflow.org/lite), for including your model
in an Android or iOS native mobile application, or in a Raspberry Pi, IoT, or
microcontroller application.
* [TensorFlow.js](https://www.tensorflow.org/js), for running your model in a
web browser or Node.JS application.
