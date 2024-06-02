from typing import Optional, Tuple

import chex
import PIL.Image
import numpy as np
from matplotlib import pyplot as plt

import sklearn.linear_model
import sklearn.metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sklearn.preprocessing
from sklearn.model_selection import train_test_split


def fig2img(fig, dpi=60) -> PIL.Image.Image:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def linear_probe_callback(features: chex.Array, labels: chex.Array, classification: bool) -> Tuple[float,float]:
    """
        Run linear probe evaluation.
        Runs logistic regression if classification is True, otherwise runs linear regression.
    """

    test_ratio = 0.1
    if labels.ndim == 1:
        labels = labels[:, None]

    if classification:
        # convert labels from float to int
        unique_labels = np.unique(labels)
        classification_labels = np.zeros(labels.shape, dtype=np.int32)
        for idx, label_it in enumerate(unique_labels):
            classification_labels[labels == label_it] = idx
        labels = classification_labels

    features = sklearn.preprocessing.normalize(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=test_ratio)  # saved two lines of code, let's go!

    if classification:
        linear_model = sklearn.linear_model.LogisticRegression(max_iter=int(1e4))
    else:
        linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(features_train, labels_train[:, 0])
    train_accuracy = linear_model.score(features_train, labels_train)
    test_accuracy = linear_model.score(features_test, labels_test)

    return train_accuracy, test_accuracy


def visualize_embeddings(mean_latents: np.ndarray, hidden_parameters: np.ndarray, do_tsne: bool) -> PIL.Image.Image:
    latent_dim = mean_latents.shape[-1]
    if latent_dim > 2:
        if do_tsne:
            tsne = TSNE()
            points = tsne.fit_transform(mean_latents)
        else:
            pca = PCA(n_components=2)
            points = pca.fit_transform(mean_latents)
    else:
        if latent_dim == 1:
            points = np.zeros([hidden_parameters.shape[0], 2])
            points[:, 0] = mean_latents.flatten()
        else:  # i.e. latent_dim == 2
            points = mean_latents
    if len(hidden_parameters.shape) == 2:
        hidden_parameters = hidden_parameters[:, 0]
    ps = plt.scatter(points[:, 0], points[:, 1], c=hidden_parameters, cmap='nipy_spectral', alpha=0.5)
    if len(np.unique(hidden_parameters)) < 10:
        plt.colorbar(ps, ticks=np.unique(hidden_parameters))
    else:
        plt.colorbar(ps)
    plt.gca().set_aspect('equal', adjustable='datalim')
    fig = plt.gcf()
    pil_img = fig2img(fig)
    # fig.canvas.draw()
    # pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.clf()
    return pil_img


