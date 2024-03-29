#!/bin/sh

# Run combinations of RL, RD, abs_dist+exp, nosofsky+squared_dist


SWEEP=(
    "game.similarity.param=0, 1, 2, 4, 6, 8, 16"
    # "simulation/dynamics=reinforcement_learning, replicator_dynamic"
    "simulation/dynamics=replicator_dynamic"
    )

ARGS=(
    "game/similarity=nosofsky_normed" 
    "game.distortion=squared_dist" 
    # "simulation.num_trials=100"
    "simulation.num_trials=8"    
    # "explore_space/pool_size=large"
    # "simulation.trajectory=True"
    "simulation.num_variants=1000"
    )

echo python src/run_simulations.py -m $"${ARGS[@]}" $"${SWEEP[@]}" 
time python src/run_simulations.py -m $"${ARGS[@]}" $"${SWEEP[@]}" 

# echo python src/explore.py $"${ARGS[@]}"
# python src/explore.py $"${ARGS[@]}"

echo python src/curve.py $"${ARGS[@]}"
python src/curve.py $"${ARGS[@]}"

# no plot, need to use notebook
