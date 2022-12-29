#!/bin/sh

SWEEP="game.similarity.param=0.1, 0.2, 0.5, 0.75, 1, 2, 3, 5, 1000"

ARGS=(
    "game/similarity=exp" 
    "game.distortion=abs_dist" 
    "simulation.num_trials=100" 
    "simulation.dynamics.num_rounds=1e5"
    "explore_space/pool_size=large"
    )

echo python src/run_simulations.py -m $"${ARGS[@]}" "${SWEEP}"
python src/run_simulations.py -m $"${ARGS[@]}" "${SWEEP}"

echo python src/explore.py $"${ARGS[@]}"
python src/explore.py $"${ARGS[@]}"

echo python src/explore.py $"${ARGS[@]}"
python src/curve.py $"${ARGS[@]}"

echo python src/explore.py $"${ARGS[@]}"
python3 src/plot.py $"${ARGS[@]}"
