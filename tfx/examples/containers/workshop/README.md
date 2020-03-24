# Workshop / Examples Container

This is a Docker image for running the
[TFX Developer Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/workshop).
It includes TensorFlow and TFX, and initializes a clean, basic environment for
running the workshop.

## Prequisites

* Mac or Linux (Highly recommended)
* Windows (optional)
* Docker
* Git
* At least 3GB available disk space

## To Run

```bash
git clone https://github.com/tensorflow/workshops.git
cd workshops/tfx_airflow
source start_container.sh
```

Note: Instructions for Windows are TBD.

## Running the [TFX Developer Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/workshop)

Follow the instructions in the tutorial, which assume that you are running on
your host, and *make a few changes to commands* which are required because you
are running inside a container instead of directly on your machine.

*Do not start Airflow webserver, Airflow scheduler, or Jupyter notebook. Those
have already been started for you in your container.*

The Airflow console will be available on the host system (in other words,
outside the container) at:

`http://localhost:8080`

Jupyter Notebook will be available on the host system at:

`http://localhost:8888`

## Using Bash (optional)

Once the container startup is complete you will be in a bash shell, which will
be logging Airflow messages.  Hit return to get to a bash prompt, or start
another shell in a different terminal. To run another shell, move to where you
cloned the workshops repo and run:

`source utils/run_bash.sh`

Since the workshop runs in a Python virtual environment you may also need to
activate that environment in the bash shell that you're running inside the
container.  It is located in the container at /root/tfx_env.

`source /root/tfx_env/bin/activate`

To exit your shells, enter `exit`.
