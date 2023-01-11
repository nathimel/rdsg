# RDSG: Rate-Distortion in Sim-max Games

This repository contains code for constructing sim-max games, simulating  dynamics, and quantifying the evolved signaling languages' information-theoretic efficiency.

The codebase is organized around the following steps of the experiment.

## Setting up an experiment

There are a number of important parameters to configure, including:

- Game size (number of states, signals)
- Distortion measure (e.g. squared distance, absolute difference)
- Similarity function (see the [perception](src/game/perception.py) submodule)
- (Im)precision parameter input to the similarity function
- Adaptive [dynamics](src/simulation/dynamics.py) for modeling evolution (replicator dynamic vs. reinforcement learning)
- How many trials to run and for how long

This codebase uses [hydra](https://hydra.cc/) to organize configurations and outputs:

- The [conf](./conf/) folder contains the main `config.yaml` file, which can be overriden with additional YAML files or command-line arguments.

- Running the shell script [run.sh](scripts/run.sh) will generate folders and files in [outputs](outputs), where a `.hydra` folder will be found with a `config.yaml` file. Reference this file as an exhaustive list of config fields to override.

## Requirements  

Step 1. Create the conda environment:

- Get the required packages by running

    `conda env create -f environment.yml`

Step 2. Install ALTK via git:

- Additionally, this project requires [the artificial language toolkit (ALTK)](https://github.com/nathimel/altk). Install it via git with

    `python3 -m pip install git+https://github.com/nathimel/altk.git`

## Replicating the experimental results

The main experimental results can be reproduced by running `./scripts/run_main_experiment.sh`.

This will perform four basic steps by running the following scripts:

1. Simulate evolution:

    `python3 src/run_simulations.py`

    Run one or more trials of an evolutionary dynamics simulation on a sim-max game, and save the resulting (rate, distortion) points to a csv.

    - This script will also generate and save hypothetical variants of the emergent systems for comparison.

2. Estimate Pareto frontier

    `python3 src/curve.py`

    Estimate the Pareto frontier for sim-max languages balancing simplicity/informativeness trade-off, which is a Rate-Distortion curve computed by the Blahut-Arimoto algorithm.

3. (Optional) Explore the trade-off space

    `python3 src/explore.py`

    Use a genetic algorithm to explore the space of possible sim-max languages.

    - This step is optional in the sense hypothetically possible languages are already generated from the emergent systems. It shows, unlike the emergent variants, that simple, uninformative systems are a very high-density region of the space of possible systems. The emergent variants explore more regions of the space.

4. Get a basic plot

    `python3 src/plot.py`

    Produce a basic plot of the emergent and explored systems compared to the Pareto frontier of optimal solutions.

    Code for the more detailed plots from the paper can be found in [notebooks/paper_figures.ipynb](src/notebooks/paper_figures.ipynb).
