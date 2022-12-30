#!/bin/sh

# Run combinations of RL, RD, abs_dist+exp, nosofsky+squared_dist


SWEEP=(
    "game.similarity.param=0.1, 0.2, 0.5, 0.6, 0.75, 1, 2, 3, 5, 1000"
    "simulation/dynamics=reinforcement_learning, replicator_dynamic"
    )

ARGS=(
    "game/similarity=exp_normed" 
    "game.distortion=abs_dist" 
    "simulation.num_trials=100"
    "explore_space/pool_size=large"
    )

# echo python src/run_simulations.py -m $"${ARGS[@]}" $"${SWEEP[@]}" 
# python src/run_simulations.py -m $"${ARGS[@]}" $"${SWEEP[@]}" 

# echo python src/explore.py $"${ARGS[@]}"
# python src/explore.py $"${ARGS[@]}"

echo python src/curve.py $"${ARGS[@]}"
python src/curve.py $"${ARGS[@]}"

# no plot, need to use notebook
