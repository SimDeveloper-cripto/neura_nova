# neura_nova/predict.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance


def load_custom_images(folder_path):
    images    = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(folder_path, filename)

                img = Image.open(img_path).convert('L')
                img = img.resize((28, 28), Image.Resampling.LANCZOS)

                img_array = np.array(img).astype(np.uint8)
                img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

                img = Image.fromarray(img_array)
                img = ImageEnhance.Sharpness(img).enhance(2.5)
                img = ImageEnhance.Contrast(img).enhance(2.0)
                img = ImageEnhance.Brightness(img).enhance(1.2)

                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = img_array.reshape(1, -1)

                images.append(img_array)
                filenames.append(filename)
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {filename}: {e}")

    if images:
        return np.vstack(images), filenames
    else:
        return np.array([]), []

def predict_custom_images(model, batch_folder_path):
    X_custom, filenames = load_custom_images(batch_folder_path)

    if len(X_custom) == 0:
        print("Nessuna immagine trovata nella cartella.")
        return

    X_custom_T  = X_custom.T
    predictions = model.predict(X_custom_T)

    plt.figure(figsize=(15, 5))
    for i in range(min(10, len(X_custom))):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_custom[i].reshape(28, 28), cmap='gray')
        predicted_class = np.argmax(predictions[:, i])
        plt.title(f"Pred: {predicted_class}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('custom_predictions.png')
    plt.show()

    for i, filename in enumerate(filenames):
        predicted_class = np.argmax(predictions[:, i])
        confidence      = predictions[predicted_class, i]
        confidence      = max(0, min(1, confidence))
    return predictions, filenames