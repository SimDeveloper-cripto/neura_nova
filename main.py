# main.py

from neura_nova import build_and_train_model_with_sgd
# from neura_nova import build_and_train_model_with_adam
# from neura_nova import build_and_train_model_with_rprop


# TODO: ADD ENCAPSULATION
# TODO: RIMUOVI CALCOLI DOPPI O MULTIPLI, SE POSSONO ESSERE FATTI SOLO UNA VOLTA FALLO (OVERHEAD)
# PROBLEM: PATTERN RECOGNITION (DISCRIMINATIVE NETWORK)
if __name__ == "__main__":
    build_and_train_model_with_sgd()