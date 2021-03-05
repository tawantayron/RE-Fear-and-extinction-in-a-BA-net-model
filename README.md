## Introduction

This is a reference implementation of the following model:

  Vlachos, I., Herry, C., Lüthi, A., Aertsen, A., & Kumar, A. "Context-dependent
  encoding of fear and extinction memories in a large-scale network model of the basal
  amygdala". PLoS Comput Biol, v. 7, n. 3, p. e1001104, 2011.

## Platform information

**Platform:** Ubuntu 20.04.2 LTS

**gcc (GCC):** 9.3.0

**Python:** 3.8.5 

**Brian 2:** 2.4.2 

**Matplotlib:** 3.3.4

**NumPy:** 1.20.1

**SciPy:** 1.6.1

In this work we replicate two models following the reference article: the first is a
mean-field model and the second is a spiking network model. Both are written in Python
and use Brian2 as the main package. To process the data and generate figures we use Numpy,
Scipy and Matplotlib. We use Multiprocessing (Python internal package) to run several simulations in parallel.

## Packages installation

**To assure that all data and figures will be generated as presented in the article, we
recommend following the instructions below and use the modules in the same versions described
here, although the code also works in newer versions.**

### Python installation
The network simulation is implemented with Python (v. 3.8.5).

To install Python 3, type in console:

```
$sudo apt-get update 
$sudo apt-get install python3.8
```

### Installing pip

We use pip, a package management system, to install the Python modules described above.
To install pip in Ubuntu type in a terminal:

```
$sudo apt install python3-pip
```

Upgrade pip to the latest version:

```
$pip3 install --upgrade pip
```

Installation of packages using pip can be done with the following command:

```
$pip3 install --user PYTHON_PACKAGE_NAME
```

#### Python modules installation using pip (recommended)

To install the required packages type in terminal:

```
$pip3 install --user brian2
$pip3 install --user matplotlib==3.3.4
$pip3 install --user numpy==1.20.1
$pip3 install --user scipy==1.6.1
```
or

```
$pip3 install --user brian2 matplotlib==3.3.4 numpy==1.20.1 scipy==1.6.1
```

All software packages are also available in the Anaconda distribution and in
brian-team channel (see below).

### Alternative installation (using Anaconda)

Alternatively, you can install the scientific Python modules with the Anaconda data science platform.

For Linux, Anaconda with Python 3.8 is only available for 64-bit systems. Download link:
https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

To install open a terminal in the folder containing the downloaded file and type:

```
$chmod +x Anaconda3-2020.11-Linux-x86_64.sh

$./Anaconda3-2020.11-Linux-x86_64.sh
```


#### Python modules installation using Anaconda

Brian 2 is not included in Anaconda by default.

To install brian-team channel, necessary to install Brian2, open a terminal and type:

```
$conda install -c conda-forge brian2
```

For further information access the website: https://brian2.readthedocs.io/en/stable/introduction/install.html


The other useful packages can be installed with:

```
$conda install PYTHON_PACKAGE_NAME
```

## Code repository

This folder contains five Python codes:
  *  **amygdala.py:** This code contains the main function to create the spiking network structure and the synaptic and external input connections. It also contains the auxiliary functions to calculate the synaptic weight normalization constant and to smooth curves in graphs.
  *  **models_eq.py:** Equations for the neuron, synapse and synaptic plasticity models.
  *  **params.py:** Main parameter values.
  *  **runsim.py:** Definition of the protocols used to reproduce the results for the spiking model.
  *  **mean_field_model.py:** Script to simulate and run the mean-field model.

## Running the scripts

### Mean-Field model

To run the simulation of the mean-field model, simply open the terminal in the folder containing the codes and execute the command:

```
$python3 mean_field_model.py
```

### Spiking network model

The main script used to simulate the network and generate figures, can be run by typing the following command in a terminal:

```
$python3 runsim.py ARG
```

where ARG is the command-line argument passed to the script to specify the protocol for the simulations.
The protocols are specified below:

  * protocol = 0:   Simulation of spontaneous network activity (Figure 4).
  * protocol = 1:   Dynamics of conditioning and extinction processes (Figures 5 and 6).
  * protocol = 2:   Fear renewal (Figures 7 and 8).
  * protocol = 3:   Gamma oscillations for high network connectivity (Figure 9).
  * protocol = 4:   Effects of connectivity, synaptic weights and delays of the inhibitory population on synchronization (Figures 10 and 11).
  * protocol = 5:   Blockage of inhibition (Figure 12 and 13).


After each simulation, the results are found in folders named with the desired protocol.

**WARNING**: About 2 GB of RAM and 3 cores are required to run protocols 1, 2, 4 and 5.

For example, to execute the "Dynamics of conditioning and extinction processes" protocol, open the folder containing the codes and type in the terminal:

```
$python3 runsim.py 1
```
