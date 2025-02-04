# main.py

# import pstats
# import cProfile
from neura_nova import run_ff_model, run_cnn_model

# PROBLEM: PATTERN RECOGNITION (DISCRIMINATIVE NETWORK)
if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    run_ff_model()
    # build_and_train_cnn_model()

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20)