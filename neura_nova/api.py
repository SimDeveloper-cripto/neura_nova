# neura_nova/api.py

from .config import load_config, update_config_results
from .ff.setup import build_and_train_ff_model_with_config
from .cnn.setup import build_and_train_cnn_model_with_config

# TODO: USE GPU TO COMPUTE
# TODO: CREARE 10 CONFIGURAZIONI PER LA FF E CNN

# TODO [SIMONE]: RISOLVERE PER BENE L'EARLY STOPPING
# TODO [CARMINE]: __REGOLA DI DECISIONE__
# TODO [CARMINE]: RIFARE DA CAPO LA RETE CONVOLUZIONALE

# TODO [PROF]: ABBIAMO PENSATO AD UN DROP-OUT PER AUMENTARE LA GENERALIZZAZIONE
# TODO [PROF]: ABBIAMO PENSATO AD UN WEIGHT-DECAY (TERMINE DI PENALITA' NELLA LOSS) NELL'AGGIORNAMENTO DEI PESI
# TODO: unificare in un unico layer convolutivo conv2d e maxpoollayer
# TODO: CNN TROPPO LENTA


# TODO: VERIFICA STESSO RAGIONAMENTO PER MAXPOOLLAYER (ks = 2, stride = 2)
# TODO: LO SPESSORE DEL FEATURE VOLUME RISULTANTE E' PARI AL NUMERO DI FILTRI? (LAYER DI POOLING)


def run_ff_model():
    configs = load_config('config/ffconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[FEED-FORWARD] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_ff_model_with_config(config)
        results.append(result)
        index += 1
    update_config_results(results, 'results/ff/results.json')

def run_cnn_model():
    configs = load_config('config/cnnconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[CONVOLUTIONAL] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_cnn_model_with_config(config)
        results.append(result)
        index += 1
    update_config_results(results, 'results/cnn/results.json')
