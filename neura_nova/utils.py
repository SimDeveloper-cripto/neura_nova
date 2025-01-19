# neura_nova/utils.py

"""
Here we encapsulate utility functions.
"""

import matplotlib.pyplot as plt


# TODO: DEFINE "mean" and "std deviation"
def show_image(batch, index):
    """
    It allows us to visualize an image from the batch, once de-normalized.

    Returns:
        None
    """

    mean = 0.1307
    std  = 0.3081

    images, labels = batch
    img = images[index]

    # De-normalization: returns the pixels to their original values
    img = img * std + mean

    img = img.squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {labels[index].item()}')
    plt.axis('off')
    plt.show()
