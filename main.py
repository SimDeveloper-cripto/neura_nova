# main.py

from neura_nova import run_ff_model, run_cnn_model

if __name__ == "__main__":
    while True:
        try:
            model = int(input("[1] FEED-FORWARD, [2] CONVOLUTIONAL: "))
            if model in (1, 2):
                break
            else:
                print("The input is not valid.")
        except ValueError:
            print("The input is not valid. Please enter a number.")

    if model == 1:
        run_ff_model()
    else:
        run_cnn_model()
