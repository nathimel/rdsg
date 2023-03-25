#!/bin/sh

SWEEP=(
    "game.similarity.param=0, 1, 2, 4, 6, 8, 16"
    # "simulation/dynamics=spillover_learning"
    # "simulation/dynamics=spillover_learning, reinforcement_learning"    
    "simulation/dynamics=noisy_replicator_dynamic, replicator_dynamic"
    # "simulation/dynamics=noisy_replicator_dynamic, replicator_dynamic, spillover_learning, reinforcement_learning"
    # "simulation/dynamics=replicator_dynamic"
    )

ARGS=(
    "game.size.num_states=10"
    "game.size.num_signals=10"
    "game/similarity=nosofsky_normed" 
    "game.distortion=squared_dist" 
    "game.prior='random'"
    "simulation.num_trials=6"
    # "explore_space/pool_size=large"
    "simulation.trajectory=True"
    "simulation.num_variants=1000"
    )

echo python src/run_simulations.py -m $"${ARGS[@]}" $"${SWEEP[@]}" 
time python src/run_simulations.py -m $"${ARGS[@]}" $"${SWEEP[@]}" 

# echo python src/explore.py $"${ARGS[@]}"
# python src/explore.py $"${ARGS[@]}"

echo python src/curve.py $"${ARGS[@]}"
python src/curve.py $"${ARGS[@]}"

# no plot, need to use notebook
