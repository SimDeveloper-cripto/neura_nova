# neura_nova/api.py

# TODO: USE GPU TO COMPUTE

from .config import load_config
from .ff.setup import build_and_train_ff_model_with_config, save_ff_to_json
from .cnn.setup import build_and_train_cnn_model_with_config, save_cnn_to_json

# TODO 0: METTERE NELLE VARIE CONFIG I PARAMETRI PER ADAM ANCHE SE SEMPRE GLI STESSI E PARAMETRIZZARE ANCHE QUELLO
# TODO 1: RISOLVERE I TODO NEI NETWORK (IMPLEMENTARE getAccuracy(...)) E ASSEGNARE IL VALORE NEL setup.py
# TODO 2: MEDIA ARITMENTICA SULLA BASE DEI RISULTATI DELLE VARIE K-FOLD
# TODO 3: __REGOLA DI DECISIONE__
# TODO 4: RISOLVERE PER BENE L'EARLY STOPPING
# TODO 5: RIFARE DA CAPO LA RETE CONVOLUZIONALE

# TODO [PROF]: ABBIAMO PENSATO AD UN DROP-OUT PER AUMENTARE LA GENERALIZZAZIONE
# TODO [PROF]: ABBIAMO PENSATO AD UN WEIGHT-DECAY (TERMINE DI PENALITA' NELLA LOSS) NELL'AGGIORNAMENTO DEI PESI

def run_ff_model():
    configs = load_config('config/ffconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[FEED-FORWARD] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_ff_model_with_config(config)
        results.append(result)
        index += 1
    save_ff_to_json(results)

def run_cnn_model():
    configs = load_config('config/cnnconfig.json')
    results = []

    index = 1
    for config in configs:
        print(f"\n[CONVOLUTIONAL] TRAINING MODEL WITH CONFIG {index}")
        result = build_and_train_cnn_model_with_config(config)
        results.append(result)
        index += 1
    save_cnn_to_json(results)
