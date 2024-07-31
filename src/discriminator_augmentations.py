import torch

import numpy as np


def disc_augment(image_batch):
    """
    Return augmented training arrays.

    :param image_batch: Batch of images to augment.
    :return: Augmented batch of images.
    """

    # 50% chance of flipping along axis -1
    if torch.randint(2, (1,)):
        image_batch = torch.flip(image_batch, [-1])
    # 50% chance of flipping along axis -2
    if torch.randint(2, (1,)):
        image_batch = torch.flip(image_batch, [-2])
    # 50% chance of flipping along axis -3
    if torch.randint(2, (1,)):
        image_batch = torch.flip(image_batch, [-3])

    # Random translation
    image_batch = random_translate(image_batch)

    return image_batch


def random_translate(x, ratio=0.05):
    """
    Randomly translate an image by a ratio of its size.

    :param x: Batch of images to translate.
    :param ratio: Ratio of image size to translate.
    :return: Translated batch of images.
    """

    max_shift = np.array((
        int((x.shape[-3] * ratio + 0.5)),
        int((x.shape[-2] * ratio + 0.5)),
        int((x.shape[-1] * ratio + 0.5))
    ))

    translation = tuple(np.random.randint(-max_shift, max_shift + 1, 3))

    x = torch.roll(x, shifts=translation, dims=(-3, -2, -1))  # cannot find a better solution

    return x
