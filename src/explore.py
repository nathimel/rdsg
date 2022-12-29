import hydra
import itertools
import os
import numpy as np

from tqdm import tqdm

from altk.effcomm.sampling import generate_languages
from altk.effcomm.agent import LiteralSpeaker, LiteralListener
from altk.effcomm.optimization import EvolutionaryOptimizer

from game.languages import SignalingLanguage, StateSpace, Signal, SignalMeaning, State
from game.signal_mutations import AddSignal, RemoveSignal, InterchangeSignal
from analysis.rd import information_rate, total_distortion, compute_rate_distortion
from analysis.measure import agents_to_channel
from simulation.driver import game_parameters

from misc import util


def array_to_points(arr: np.ndarray) -> list[State]:
    return [State(name=str(point)) for item in np.argwhere(arr) for point in item]


def generate_meanings(universe: StateSpace) -> list:
    """Generates all possible subsets of the meaning space."""
    arrs = [
        np.array(i)
        for i in itertools.product([0, 1], repeat=len(universe.referents))
    ]
    arrs = arrs[1:]  # remove the empty array meaning to prevent div by 0
    meanings = [SignalMeaning(array_to_points(arr), universe) for arr in arrs]
    return meanings


def generate_expressions(universe: StateSpace) -> list[Signal]:
    print(f"Generating {2 ** len(universe)} expressions...")
    meanings = generate_meanings(universe)
    expressions = [
        Signal(
            form=f"dummy_form_{i}",
            meaning=meaning,
        )
        for i, meaning in tqdm(enumerate(meanings))
    ]
    return expressions

def lang_to_cond_dist(lang: SignalingLanguage) -> np.ndarray:
    """Get P(a|s) the conditional probability distribution of acts given states, using a language (specifying only states/acts given signals) to initialize altk LiteralSpeakers, LiteralListeners."""
    s = LiteralSpeaker(lang)
    r = LiteralListener(lang)
    cond = agents_to_channel(s, r, fill_rows=True)
    return cond


##############################################################################
# Main driver code
##############################################################################

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)
    kwargs = util.experiment_parameters(config)
    game_params = game_parameters(**kwargs)

    # save one sample of points for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    save_fn = os.path.join(game_dir, config.filepaths.sampled_points_save_fn)

    # Load evolutionary optimizer params
    seed_size = kwargs["seed_gen_size"]
    max_mutations = kwargs["max_mutations"]
    generations = kwargs["num_generations"]

    universe = game_params["sender"].language.universe
    expressions = generate_expressions(universe)
    lang_size = kwargs["num_signals"]

    # Define rate-distortion point measurements
    prior = game_params["prior"]
    dist_mat = game_params["dist_mat"]

    complexity_measure = lambda lang: information_rate(
        p_x=prior,
        p_xhat_x=lang_to_cond_dist(lang),
    )

    comm_cost_measure = lambda lang: total_distortion(
        p_x=prior,
        p_xhat_x=lang_to_cond_dist(lang),
        dist_mat=dist_mat,
    )

    # We aren't interested in bottlenecks unless they emerge naturally
    if lang_size < len(universe.referents):
        raise Exception(
            "Language size must be greater than or equal to the universe size."
        )
    
    # Step 1: generate a random sample of languages
    print(seed_size)
    result = generate_languages(
        language_class=SignalingLanguage,
        expressions=expressions,
        lang_size=lang_size,
        sample_size=seed_size,
        verbose=True,
    )
    seed_population = result["languages"]
    id_start = result["id_start"]


    # Step 2: use optimizer as an exploration / sampling method:
    # estimate FOUR pareto frontiers using the evolutionary algorithm; one for each corner of the 2D space of possible langs
    directions = {
        "lower_left": ("complexity", "comm_cost"),
        "lower_right": ("simplicity", "comm_cost"),
        "upper_left": ("complexity", "informativity"),
        "upper_right": ("simplicity", "informativity"),
    }
    objectives = {
        "comm_cost": comm_cost_measure,
        "informativity": lambda lang: -1 * comm_cost_measure(lang),
        "complexity": complexity_measure,
        "simplicity": lambda lang: -1 * complexity_measure(lang),
    }

    # Load signal-specific mutations
    mutations = [
        AddSignal(),
        RemoveSignal(),
        InterchangeSignal(),
    ]

    # Initialize optimizer
    optimizer = EvolutionaryOptimizer(
        objectives=objectives,
        expressions=expressions,
        mutations=mutations,
        sample_size=seed_size,
        max_mutations=max_mutations,
        generations=generations,
        lang_size=lang_size,
    )


    # Explore corners of the possible language space
    results = {k: None for k in directions}
    pool = []
    for direction in directions:
        if direction not in kwargs["explore_directions"]:
            continue

        # set directions of optimization
        x, y = directions[direction]
        optimizer.x = x
        optimizer.y = y
        print(f"Optimizing for {direction} region (min {x}, {y}) ...")

        # run algorithm
        result = optimizer.fit(
            seed_population=seed_population,
            id_start=id_start,
        )

        # collect results
        results[direction] = result
        id_start = result["id_start"]
        pool.extend(results[direction]["explored_languages"])

    pool = list(set(pool))

    # Save RD values
    points = [compute_rate_distortion(prior, lang_to_cond_dist(lang), dist_mat) for lang in pool]
    print("num distinct explored points:", len(set(points)))

    util.save_points_df(save_fn, util.points_to_df(points))

if __name__ == "__main__":
    main()