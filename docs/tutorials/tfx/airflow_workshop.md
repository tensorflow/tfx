# **TFX Airflow Tutorial**

## Overview
In this lab you will learn how to:
* Create machine learning pipelines and its integration with tfx and tensorboard
* Show the interaction of tfx with jupyter notebooks
* Using airflow to execute tasks on gcp

## **TFX with airflow**
 A TFX pipeline is a Directed Acyclic Graph, or "DAG". We will often refer to pipelines as DAGs.
You'll follow a typical ML development process, starting by examining the dataset, and end up with a complete working pipeline. Along the way you'll explore ways to debug and update your pipeline, and measure performance.

Please see the [TFX User Guide](https://www.tensorflow.org/tfx/guide) to learn more.

## **Chicago Taxi Dataset**

![taxi.jpg](images/airflow_workshop/taxi.jpg)

![chicago.png](images/airflow_workshop/chicago.png)

You're using the [Taxi Trips dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) released by the City of Chicago.

### Model Goal - Binary classification
Will the customer tip more or less than 20%?

## Setup and Requirements

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
gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  notebooks.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com
  ```


## Deploy Vertex Notebook instance
1. Click on the __Navigation Menu__.
2. Navigate to __Vertex AI__, then to __Workbench__

![vertex-ai-workbench.png](images/airflow_workshop/vertex-ai-workbench.png)

3. On the Notebook instances page, navigate to the __User-Managed Notebooks__ tab and click **New Notebook**.
4. In the Customize instance menu, select **TensorFlow Enterprise** and choose the version of **TensorFlow Enterprise 2.3 (with LTS)** > **Without GPUs**.

![vertex-notebook-create-2.png](images/airflow_workshop/vertex-notebook-create-2.png)

5. In the __New notebook instance__ dialog, for __Region__, select `us-central1`, for __Zone__, select a zone within the selected region, leave all other fields with their default options, and click __Create__.

After a few minutes, the Vertex AI console will display your instance name, followed by `Open Jupyterlab`.

6. Click **Open JupyterLab**. Your notebook is now set up.

Click _Check my progress_ to verify the objective.
  <ql-activity-tracking step=1>
      Create a Vertex AI Notebook
  </ql-activity-tracking>

## Set up the environment

### Clone the lab repository
Next you'll clone the `tfx` repository in your JupyterLab instance.
1. In JupyterLab, click the __Terminal__ icon to open a new terminal.
<ql-infobox><strong>Note:</strong> If prompted, click <code>Cancel</code> for Build Recommended.</ql-infobox>
2. To clone the `tfx` Github repository, type in the following command, and press __Enter__.

```bash
git clone https://github.com/tensorflow/tfx.git
```

3. To confirm that you have cloned the repository, double-click the `training-data-analyst` directory and confirm that you can see its contents.

![repo-directory.png](images/airflow_workshop/repo-directory.png)
  
### Install lab dependencies
1. Run the following to go to the `tfx/tfx/examples/airflow_workshop/taxi/setup/` folder, then run `./setup_demo.sh` to install lab dependencies:

```bash
cd ~/tfx/tfx/examples/airflow_workshop/taxi/setup/
./setup_demo.sh
```

Click _Check my progress_ to verify the objective.
  <ql-activity-tracking step=2>
      Clone the lab repository
  </ql-activity-tracking>

### Run airflow server from your shell

1. Run the `airflow users  create` command to create an admin user for airflow. Then run the `airflow webserver` and `airflow scheduler` command to run the server

```bash
airflow users  create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin
nohup airflow webserver -p 7000 &> webserver.out &
nohup airflow scheduler &> scheduler.out &
```
### Create firewall rule to access to airflow server in browser
1. Go to `https://console.cloud.google.com/networking/firewalls/list` and make sure the project name is selected appropriately
2. Click on `CREATE FIREWALL RULE` option on top

![firewall-rule.png](images/airflow_workshop/firewall-rule.png)

3. In the **Create a firewall dialog**, for **Priority**, select `1`, for **Targets**, select `All instances in the network`,for **Source IPv4 ranges**, select `0.0.0.0/0`, for **Protocols and ports**, click on `tcp` and enter `7000` in the box next to `tcp`

![create-firewall-dialog.png](images/airflow_workshop/create-firewall-dialog.png)

#### DAG view buttons 

![dag-buttons.png](images/airflow_workshop/dag-buttons.png)

* Use the button on the left to enable the DAG
* Use the button on the right to trigger the DAG
* Click on taxi to go to the graph view of the DAG
* In the taxi DAG page use the button on the right to refresh the DAG when you make changes

#### Airflow CLI
You can also use the [Airflow CLI](https://airflow.apache.org/cli.html) in terminal to enable and trigger your DAGs:

```bash
# enable/disable
airflow pause <your DAG name>
airflow unpause <your DAG name>

# trigger
airflow trigger_dag <your DAG name>
```

### In a browser
Open a browser and go to http://<external_ip>:7000

* Enter username and password mentioned in `airflow users  create` command
* Trigger taxi

#### Waiting for the pipeline to complete
After you've triggered your pipeline in the DAGs view, you can watch as your pipeline completes processing. As each component runs the outline color of the component in the DAG graph will change to show its state. When a component has finished processing the outline will turn dark green to show that it's done.

![dag-step7.png](images/airflow_workshop/dag-step7.png)

Click _Check my progress_ to verify the objective.
  <ql-activity-tracking step=3>
      Clone the lab repository
  </ql-activity-tracking>


## Understanding the components
Here we will look at the components of pipeline in detail and individually look at the outputs produced by each step in the pipeline

* Go to `~/tfx/tfx/examples/airflow_workshop/taxi/setup/` and open `notebook.ipynb`
* Follow the notebook

Click _Check my progress_ to verify the objective.
  <ql-activity-tracking step=4>
      Clone the lab repository
  </ql-activity-tracking>
