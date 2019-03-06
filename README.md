This repository contains recipes to create
[Singularity](https://www.sylabs.io/singularity/) containers which include
[Anaconda Python Distribution](https://www.anaconda.com/distribution/) as well
as other key data analysis packages.

# Building the Containers

Building the Singularity containers in this repository consists of two steps,
generating the recipes and the builds themselves.  

To generate the container recipes run the command
`../create_singularity_recipe.py`.  This scripts create a set of six
*Singularity.{cpu/gpu_cuda80/gpu_cuda90}_{27/36}* recipe files which are
generating by filling in the appropriate variables in the recipe template
*singularity.template* for the various combinations of configurations.  This
script can also be used to generate a single recipe by specifying the
`--python_ver` and `--env_name arguments`.  Use the `--help` argument to show
information about these and other arguments.  

Each recipe file specifies the steps needed create a conda environment
containing a recent version, 2018.12, of Anaconda as well some additional data
analysis packages including Tensorflow, Caffe and PyTorch.  The difference
between the six recipe is the version of Python they install, 2.7 or 3.6, if
they install GPU accelerated versions of the packages, cpu vs gpu, and for the
GPU accelerated recipes which version of CUDA is installed, 8.0 or 9.0.  

Once the recipe have been generated Singularity containers can be created by
running `sudo singularity build container_name recipe_name` where
container_name and recipe_name should be replaced by the name of the container
and the filename of the recipes.  For example to build a container named
`container_gpu_cuda80_36.simg` from the recipes Python 3.6 GPU accelerated
packages and CUDA 8.0 run: `sudo singularity build container_gpu_cuda80_36.simg
Singularity.gpu_cuda80_36`.

The resulting container can be run by executing the created file or using the
`singularity run` command.  For GPU accelerated containers the `--nv` argument
should be added to to singularity command to enable NVIDIA support in the
container.

The containers in this repository are also built on 
[Singularity Hub](https://www.singularity-hub.org/collections/2250).

# Testing Containers

A test suite is included in this repository which can be used to test basic
functionality of installed libraries in the containers.  These tests should be
run inside the containers after they have built to verify their functionality.
To run these tests start the container using `singularity run
container_name.simg`.  From the shell execute `pytest tests/test_cpu*`. 

For GPU containers start the container with using `singularity run --nv
container_name.simg` and execute the test using `pytest tests/test*`

# Logging Imports

The Singularity containers created via these means will log imports of Python
libraries to a file or a remote server if certain environment variables are
set.  

The **PYIMPORT_LIBLIST** variable should be set to a comma separated string of
the libraries whose imports are to be logged.  For example to log all imports
of NumPy and Tensorflow use `PYIMPORT_LIBLIST=”numpy,tensorflow”`.

These imports will be logged to the file specified by the **PYIMPORT_LOGFILE**
variable.  If this variable is not set imports will not be logged to a file.
When setting this file is may be desirable to take the username or date into
account to create logs for each user or a particular date range.

The **PYIMPORT_HOST** and **PYIMPORT_PORT** variables can set used to set a
hostname or IP and port for a server which will receive UDP logging messages
from the containers.  The network must be configured to allow the container to
send UDP packets to the specified server.  The *listen.py* script provides a
simple example of how these messages can be captured and decoded.  If
**PYIMPORT_HOST** or **PYIMPORT_PORT** are not set no logging to a remote
server is performed.

The format of the message which is logged to the log file and the remote server
can be controlled by setting the **PYIMPORT_MSGFORMAT** variable to a string.
The username or name of the import can be including by including {username} or
{import_name} in the string.  For example to record a log message like “jhelmus
imported numpy” use `PYIMPORT_MSGFORMAT=”{username} imported {import_name}”`.
These message are prefixed by a timestamp when they are recorded in the log
file

To aid in debugging if the **PYIMPORT_DEBUG** variable is set to 1 the import
logs are printed to stderr in addition the other logging methods.
