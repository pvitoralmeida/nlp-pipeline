import random
from itertools import product


def generate_all_combinations(config_space):
    keys = config_space.keys()
    values = config_space.values()

    for combination in product(*values):
        yield dict(zip(keys, combination))


def sample_random_configs(config_space, n_samples):
    all_configs = list(generate_all_combinations(config_space))
    return random.sample(all_configs, min(n_samples, len(all_configs)))