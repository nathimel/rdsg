#!/bin/sh

# TODO: add command line args
python src/run_simulations.py

python src/explore.py

python src/curve.py

python3 src/plot.py
