# SUEP Coffea Dask

Repository for SUEP using fastjet with awkward input from PFnano nanoAOD samples. To run copy the runner.py into the main repo directory and run the following.
We also have notebooks for quick Dask studies.

## Structure

Example worfkflow for SUEP is included.

Each workflow can be a separate "processor" file, creating the mapping from NanoAOD to
the histograms we need. Workflow processors can be passed to the `runner.py` script
along with the fileset these should run over. Multiple executors can be chosen
(for now iterative - one by one, uproot/futures - multiprocessing and dask-slurm).

To run the example, run:

```
python runner.py --workflow SUEP
```

Example plots can be found in ` make_some_plots.ipynb` though we might want to make
that more automatic in the end.

## Requirements

### Coffea installation with Miniconda

For installing Miniconda, see also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Local-or-remote

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Run and follow instructions on screen
bash Miniconda3-latest-Linux-x86_64.sh
```

NOTE: always make sure that conda, python, and pip point to local Miniconda installation (`which conda` etc.).

You can either use the default environment`base` or create a new one:

```
# create new environment with python 3.7, e.g. environment of name `coffea`
conda create --name coffea python=3.7
# activate environment `coffea`
conda activate coffea
```

Install coffea, xrootd, and more. SUEP analysis uses Fastjet with awkward array input (fastjet>=3.3.4.0rc8) and vector:

```
pip install git+https://github.com/CoffeaTeam/coffea.git #latest published release with `pip install coffea`
conda install -c conda-forge xrootd
conda install -c conda-forge ca-certificates
conda install -c conda-forge ca-policy-lcg
conda install -c conda-forge dask-jobqueue
conda install -c anaconda bokeh
conda install -c conda-forge 'fsspec>=0.3.3'
conda install dask
conda install pytables
pip install --pre fastjet
pip install vector
```

For work at the LPC or coffea-casa the fastjet package is already included in the relevant singularity image and it's not required to install it in the local environment (see below)

### Other installation options for coffea

See https://coffeateam.github.io/coffea/installation.html

### Running jupyter remotely

See also https://hackmd.io/GkiNxag0TUmHnnCiqdND1Q#Remote-jupyter

1. On your local machine, edit `.ssh/config`:

```
Host lxplus*
  HostName lxplus7.cern.ch
  User <your-user-name>
  ForwardX11 yes
  ForwardAgent yes
  ForwardX11Trusted yes
Host *_f
  LocalForward localhost:8800 localhost:8800
  ExitOnForwardFailure yes
```

2. Connect to remote with `ssh lxplus_f`
3. Start a jupyter notebook:

```
jupyter notebook --ip=127.0.0.1 --port 8800 --no-browser
```

4. URL for notebook will be printed, copy and open in local browser

## Scale-out (Sites)

Scale out can be notoriously tricky between different sites. Coffea's integration of `slurm` and `dask`
makes this quite a bit easier and for some sites the ``native'' implementation is sufficient, e.g Condor@DESY.
However, some sites have certain restrictions for various reasons, in particular Condor @CERN and @FNAL.

### Condor@FNAL (CMSLPC)

The fastjet package is already included in the relevant singularity image and it's not required to install it in the local environment

Follow setup instructions at https://github.com/CoffeaTeam/lpcjobqueue. After starting
the singularity container run with

```bash
python runner.py --wf SUEP --executor dask/lpc --isMC=1 --era=2018
```

### Condor@CERN (lxplus)

Only one port is available per node, so its possible one has to try different nodes until hitting
one with `8786` being open. Other than that, no additional configurations should be necessary.

```bash
python runner.py --wf SUEP --executor dask/lxplus --isMC=1 --era=2018
```

### Coffea-casa (Nebraska AF)

The fastjet package is already included in the relevant singularity image and it's not required to install it in the local environment

Coffea-casa is a JupyterHub based analysis-facility hosted at Nebraska. For more information and setup instructions see
https://coffea-casa.readthedocs.io/en/latest/cc_user.html

After setting up and checking out this repository (either via the online terminal or git widget utility run with

```bash
python runner.py --wf SUEP --executor dask/casa --isMC=1 --era=2018
```

Authentication is handled automatically via login auth token instead of a proxy. File paths need to replace xrootd redirector with "xcache", `runner.py` does this automatically.

### Slurm@MIT (submit)

After setting up and checking out this repository (either via the online terminal or git widget utility run with

```bash
python runner.py --wf SUEP --executor dask/slurm --isMC=1 --era=2018
```

uses 'dashboard_address': 8000
ssh -L 8000:localhost:8000 <uname>@submit.mit.edu

http://localhost:8000/status
