# main.py

# import pstats
# import cProfile
from neura_nova import build_and_train_models

# PROBLEM: PATTERN RECOGNITION (DISCRIMINATIVE NETWORK)
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    build_and_train_models()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20)