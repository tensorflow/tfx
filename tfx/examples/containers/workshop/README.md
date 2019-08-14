# Workshop / Examples Container

This is a Docker image for running the
[TFX Developer Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/workshop)
and other similar examples.  It does not load TensorFlow or TFX, but only
initializes a clean, basic environment for cloning, installing, and running the
workshop and examples themselves, and includes the basic dependencies.

## Prequisites

* Docker
* At least 2GB available disk space

## To Run

Note: You should create the `airflow` directory BEFORE running
`docker-compose up`, and edit the `AIRFLOW_PATH_ON_YOUR_HOST` placeholder in
docker-compose.yml to point to where you created the directory on your system.
In the container, it will be mounted as `/home/tfx_user/airflow`.

Note: You should clone the TFX repo on your system:

```bash
git clone https://github.com/tensorflow/tfx.git
cd tfx
git checkout -f origin/r0.13
```

Edit the `TFX_PATH_ON_YOU_HOST` placeholder in docker-compose.yml to point to
where you cloned the repo on your system. In the container, it will be mounted
as `/home/tfx_user/tfx`.

Then run:

`docker-compose run --service-ports tfx`

That will build the image, create a container and run it, and log you in as
`tfx_user` (password "tfx") in the current shell on your host machine.

## Running the [TFX Developer Tutorial](https://www.tensorflow.org/tfx/tutorials/tfx/workshop)

Follow the instructions in the tutorial, which assume that you are running on
your host, and make a few changes to commands which are required because you
are running inside a container instead.

The Airflow console will be available on the host system at:

`http://localhost:8080`

Jupyter Notebook will be available on the host system at:

`http://localhost:8888`

Note: To start Jupyter Notebook in the container, run:

`jupyter notebook --ip=0.0.0.0 --no-browser`

## Using Bash

When you first run `docker-compose`, after the build finishes you will be in a
bash shell in the container, as `tfx_user` (which has sudo privileges).  To run
another shell, you first need to get the name of the container with `docker ps`.
Then, assuming that the name of the container is "tfx_docker", you would run:

`docker exec -it tfx_docker bash`

Using additional shells you can start Airflow's webserver and scheduler, and/or
Jupyter Notebook (see command syntax above).  Don't forget to also activate the
Python virtual environment that you create in the workshop.

To exit your shells, enter `exit`.
