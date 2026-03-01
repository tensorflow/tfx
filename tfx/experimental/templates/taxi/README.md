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


## Content of the template

The template generates three different kinds of python code which are needed to
run a TFX pipeline.

- `pipeline` directory contains *a pipeline definition* which specify what
  components to use and how the components will be interconnected.
  And it also contains various config variables to run the pipeline.
- `models` directory contains ML model definitions which is required by
  `Trainer`, `Transform` and `Tuner` components.
- The last piece is a platform specific configuration that describes physical
  paths and orchestrators. Currently we have `beam_dag_runner.py` and
  `kubeflow_dag_runner.py`. These files are located at the root of the project
  directory.
