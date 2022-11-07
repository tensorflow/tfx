# **TFX Airflow Tutorial**

## Overview

## Overview

This tutorial is designed to help you learn to create your own machine learning pipelines using TensorFlow Extended (TFX) and Apache Airflow as the orchestrator. It runs on on Vertex AI Workbench, and shows integration with TFX and TensorBoard as well as interaction with TFX in a Jupyter Lab environment.

### What you'll be doing?
You’ll learn how to create an ML pipeline using TFX

* A TFX pipeline is a Directed Acyclic Graph, or "DAG". We will often refer to pipelines as DAGs.
* TFX pipelines are appropriate when you will be deploying a production ML application
* TFX pipelines are appropriate when datasets are large, or may grow to be large
* TFX pipelines are appropriate when training/serving consistency is important
* TFX pipelines are appropriate when version management for inference is important
* Google uses TFX pipelines for production ML

Please see the [TFX User Guide](https://www.tensorflow.org/tfx/guide) to learn more.

You'll follow a typical ML development process:

* Ingesting, understanding, and cleaning our data
* Feature engineering
* Training
* Analyzing model performance
* Lather, rinse, repeat
* Ready for production 

## **Apache Airflow for Pipeline Orchestration**
TFX orchestrators are responsible for scheduling components of the TFX pipeline based on the dependencies defined by the pipeline. TFX is designed to be portable to multiple environments and orchestration frameworks. One of the default orchestrators supported by TFX is [Apache Airflow](https://www.tensorflow.org/tfx/guide/airflow). 
This lab illustrates the use of Apache Airflow for TFX pipeline orchestration. Apache Airflow is a platform to programmatically author, schedule and monitor workflows. TFX uses Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The rich user interface makes it easy to visualize pipelines running in production, monitor progress, and troubleshoot issues when needed. Apache Airflow workflows are defined as code. This makes them more maintainable, versionable, testable, and collaborative. Apache Airflow is suited for batch processing pipelines. It is lightweight and easy to learn.

In this example, we are going to run a TFX pipeline on an instance by manually setting up Airflow.

The other default orchestrators supported by TFX are Apache Beam and Kubeflow. [Apache Beam](https://www.tensorflow.org/tfx/guide/beam_orchestrator) can run on multiple data processing backends (Beam Ruunners). Cloud Dataflow is one such beam runner which can be used for running TFX pipelines. Apache Beam can be used for both streaming and batch processing pipelines.    
[Kubeflow](https://www.tensorflow.org/tfx/guide/kubeflow) is an open source ML platform dedicated to making deployments of machine learning (ML) workflows on Kubernetes simple, portable and scalable. Kubeflow can be used as an orchestrator for TFFX pipelines when they need to be deployed on Kubernetes clusters. 
In addition, you can also use your own [custom orchestrator](https://www.tensorflow.org/tfx/guide/custom_orchestrator) to run a TFX pipeline.

Read more about Airflow [here](https://airflow.apache.org/).

## **Chicago Taxi Dataset**

![taxi.jpg](images/airflow_workshop/taxi.jpg)

![chicago.png](images/airflow_workshop/chicago.png)

You'll be using the [Taxi Trips dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) released by the City of Chicago.

Note: This tutorial builds an application using data that has been modified for use from its original source, www.cityofchicago.org, the official website of the City of Chicago. The City of Chicago makes no claims as to the content, accuracy, timeliness, or completeness of any of the data provided at in this tutorial. The data provided at this site is subject to change at any time. It is understood that the data provided in this tutorial is being used at one’s own risk.

### Model Goal - Binary classification
Will the customer tip more or less than 20%?

## Setup the Google Cloud Project

**Before you click the Start Lab button**
Read these instructions. Labs are timed and you cannot pause them. The timer, which starts when you click **Start Lab**, shows how long Google Cloud resources will be made available to you.

This hands-on lab lets you do the lab activities yourself in a real cloud environment, not in a simulation or demo environment. It does so by giving you new, temporary credentials that you use to sign in and access Google Cloud for the duration of the lab.

**What you need**
To complete this lab, you need:

* Access to a standard internet browser (Chrome browser recommended).
* Time to complete the lab.

**Note:** If you already have your own personal Google Cloud account or project, do not use it for this lab.

**Note:** If you are using a Chrome OS device, open an Incognito window to run this lab.

**How to start your lab and sign in to the Google Cloud Console**
1. Click the **Start Lab** button. If you need to pay for the lab, a pop-up opens for you to select your payment method. On the left is a panel populated with the temporary credentials that you must use for this lab.

![qwiksetup1.png](images/airflow_workshop/qwiksetup1.png)

2. Copy the username, and then click **Open Google Console**. The lab spins up resources, and then opens another tab that shows the **Sign in** page.

![qwiksetup2.png](images/airflow_workshop/qwiksetup2.png)

_**Tip:**_ Open the tabs in separate windows, side-by-side.

![qwiksetup3.png](images/airflow_workshop/qwiksetup3.png)

3. In the **Sign in** page, paste the username that you copied from the left panel. Then copy and paste the password.

_**Important:**_- You must use the credentials from the left panel. Do not use your Google Cloud Training credentials. If you have your own Google Cloud account, do not use it for this lab (avoids incurring charges).

4. Click through the subsequent pages:
* Accept the terms and conditions.

* Do not add recovery options or two-factor authentication (because this is a temporary account).

* Do not sign up for free trials.

After a few moments, the Cloud Console opens in this tab.

**Note:** You can view the menu with a list of Google Cloud Products and Services by clicking the **Navigation menu** at the top-left.

![qwiksetup4.png](images/airflow_workshop/qwiksetup4.png)

### Activate Cloud Shell
Cloud Shell is a virtual machine that is loaded with development tools. It offers a persistent 5GB home directory and runs on the Google Cloud. Cloud Shell provides command-line access to your Google Cloud resources.

In the Cloud Console, in the top right toolbar, click the **Activate Cloud Shell** button.

![qwiksetup5.png](images/airflow_workshop/qwiksetup5.png)

Click **Continue**.

![qwiksetup6.png](images/airflow_workshop/qwiksetup6.png)

It takes a few moments to provision and connect to the environment. When you are connected, you are already authenticated, and the project is set to your _PROJECT_ID_. For example:

![qwiksetup7.png](images/airflow_workshop/qwiksetup7.png)


`gcloud` is the command-line tool for Google Cloud. It comes pre-installed on Cloud Shell and supports tab-completion.

You can list the active account name with this command:

```
gcloud auth list
```
(Output)
>ACTIVE: *
ACCOUNT: student-01-xxxxxxxxxxxx@qwiklabs.net
To set the active account, run:
    $ gcloud config set account `ACCOUNT`

You can list the project ID with this command:
```
gcloud config list project
```
(Output)
>[core]
project = <project_ID>

(Example output)
>[core]
project = qwiklabs-gcp-44776a13dea667a6

For full documentation of gcloud see the [gcloud command-line tool overview](https://cloud.google.com/sdk/gcloud).

## Enable Google Cloud services
1. In Cloud Shell, use gcloud to enable the services used in the lab.
```
gcloud services enable notebooks.googleapis.com
  ```


## Deploy Vertex Notebook instance
1. Click on the __Navigation Menu__ and navigate to __Vertex AI__, then to __Workbench__.

![vertex-ai-workbench.png](images/airflow_workshop/vertex-ai-workbench.png)

2. On the Notebook instances page, click __New Notebook__.

3. In the Customize instance menu, select **TensorFlow Enterprise** and choose the version of **TensorFlow Enterprise 2.x (with LTS)** > **Without GPUs**.

![vertex-notebook-create-2.png](images/airflow_workshop/vertex-notebook-create-2.png)

4. In the __New notebook instance__ dialog, click the pencil icon to __Edit__ instance properties.

5. For __Instance name__, enter a name for your instance.

6. For __Region__, select `us-east1` and for __Zone__, select a zone within the selected region.

7. Scroll down to Machine configuration and select __n1-standard-2__ for Machine type.

8. Leave the remaining fields with their default and click __Create__.

After a few minutes, the Vertex AI console will display your instance name, followed by __Open Jupyterlab__.

9. Click __Open JupyterLab__. A JupyterLab window will open in a new tab.

## Setup the environment

### Clone the lab repository
Next you'll clone the `tfx` repository in your JupyterLab instance.
1. In JupyterLab, click the __Terminal__ icon to open a new terminal.
<ql-infobox><strong>Note:</strong> If prompted, click <code>Cancel</code> for Build Recommended.</ql-infobox>
2. To clone the `tfx` Github repository, type in the following command, and press __Enter__.

```bash
git clone https://github.com/tensorflow/tfx.git
```

3. To confirm that you have cloned the repository, double-click the `tfx` directory and confirm that you can see its contents.

![repo-directory.png](images/airflow_workshop/repo-directory.png)
  
### Install lab dependencies
1. Run the following to go to the `tfx/tfx/examples/airflow_workshop/taxi/setup/` folder, then run `./setup_demo.sh` to install lab dependencies:

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

The above code will

* Install the required packages.
* Create an `airflow` folder in the home folder.
* Copy the `dags` folder from `tfx/tfx/examples/airflow_workshop/taxi/setup/` folder to `~/airflow/` folder.
* Copy the csv file from `tfx/tfx/examples/airflow_workshop/taxi/setup/data` to `~/airflow/data`.

![airflow-home.png](images/airflow_workshop/airflow-home.png)

## Configuring Airflow server

### Create firewall rule to access to airflow server in the browser
1. Go to `https://console.cloud.google.com/networking/firewalls/list` and make sure the project name is selected appropriately
2. Click on `CREATE FIREWALL RULE` option on top

![firewall-rule.png](images/airflow_workshop/firewall-rule.png)

In the **Create a firewall dialog**, follow the steps listed below. 

1. For **Name**, put `airflow-tfx`.
2. For **Priority**, select `1`.
3. For **Targets**, select `All instances in the network`.
4. For **Source IPv4 ranges**, select `0.0.0.0/0`
5. For **Protocols and ports**, click on `tcp` and enter `7000` in the box next to `tcp`
6. Click `Create`.

![create-firewall-dialog.png](images/airflow_workshop/create-firewall-dialog.png)

### Run airflow server from your shell

In the Jupyter Lab Terminal window, change to home directory, run the `airflow users create` command to create an admin user for Airflow:

```bash
cd
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
```

Then run the `airflow webserver` and `airflow scheduler` command to run the server.  Choose port `7000` since it is allowed through firewall.

```bash
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```

### Get your external ip

1. In Cloud Shell, use `gcloud` to get the External IP.

```
gcloud compute instances list
```

![gcloud-instance-ip.png](images/airflow_workshop/gcloud-instance-ip.png)

## Running a DAG/Pipeline

### In a browser
Open a browser and go to http://<external_ip>:7000

* In the login page, enter the username(`admin`) and password(`admin`) you chose when running the `airflow users create` command.

![airflow-login.png](images/airflow_workshop/airflow-login.png)

Airflow loads DAGs from Python source files. It takes each file and executes it. Then it loads any DAG objects from that file.
All `.py` files which define DAG objects will be listed as pipelines in the airflow homepage.
 
In this tutorial, Airflow scans the `~/airflow/dags/` folder for DAG objects. 

If you open `~/airflow/dags/taxi_pipeline.py` and scroll to the bottom, you can see that it creates and stores a DAG object in a variable named `DAG`. Hence it will be listed as a pipeline in the airflow homepage as shown below:

![dag-home-full.png](images/airflow_workshop/dag-home-full.png)

If you click on taxi, you will be redirected to the grid view of the DAG. You can click the `Graph` option on top to get the graph view of the DAG.

![airflow-dag-graph.png](images/airflow_workshop/airflow-dag-graph.png)

### Trigger the taxi pipeline

On the homepage you can see the buttons that can be used to interact with the DAG.

![dag-buttons.png](images/airflow_workshop/dag-buttons.png)


Under the **actions** header, click on the **trigger** button to trigger the pipeline.

In the taxi **DAG** page, use the button on the right to refresh the state of the graph view of the DAG as the pipeline runs.
Additionally, you can enable **Auto Refresh** to instruct Airflow to automatically refresh the graph view as and when the state changes.

![dag-button-refresh.png](images/airflow_workshop/dag-button-refresh.png)

You can also use the [Airflow CLI](https://airflow.apache.org/cli.html) in the terminal to enable and trigger your DAGs:

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

#### Waiting for the pipeline to complete
After you've triggered your pipeline, in the DAGs view, you can watch the progress of your pipeline while it is running. As each component runs, the outline color of the component in the DAG graph will change to show its state. When a component has finished processing the outline will turn dark green to show that it's done.

![dag-step7.png](images/airflow_workshop/dag-step7.png)

## Understanding the components
Now we will look at the components of this pipeline in detail, and individually look at the outputs produced by each step in the pipeline.

1. In JupyterLab go to `~/tfx/tfx/examples/airflow_workshop/taxi/notebooks/`

2. Open **notebook.ipynb.**
![notebook-ipynb.png](images/airflow_workshop/notebook-ipynb.png)

3. Continue the lab in the notebook, and run each cell by clicking the **Run** ( <img src="images/airflow_workshop/f1abc657d9d2845c.png" alt="run-button.png"  width="28.00" />) icon at the top of the screen. Alternatively, you can execute the code in a cell with **SHIFT + ENTER**.

Read the narrative and make sure you understand what's happening in each cell.