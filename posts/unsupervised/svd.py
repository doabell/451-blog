import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

import PIL
import urllib


def url2img(url: str) -> npt.NDArray:
    """
    Downloads image from `url` and turns it into grayscale, using [CCIR 601 weights](https://en.wikipedia.org/wiki/Luma_(video)).

    The original image must **not** be grayscale.

    Args:
        url (str): link to download image from.

    Returns:
        npt.NDArray: grayscale image.
    """
    im = np.array(PIL.Image.open(urllib.request.urlopen(url)))
    return 1 - np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])


def compare_images(A: npt.NDArray, A_: npt.NDArray):
    """
    Plots two grayscale images, `A` and `A_`, side by side.

    For best results, the images should be the same size.

    Args:
        A (npt.NDArray): grayscale image to compare
        A_ (npt.NDArray): grayscale image to compare
    """

    fig, axarr = plt.subplots(1, 2, figsize=(7, 3))

    axarr[0].imshow(A, cmap="Greys")
    axarr[0].axis("off")
    axarr[0].set(title="Original image")

    axarr[1].imshow(A_, cmap="Greys")
    axarr[1].axis("off")
    axarr[1].set(title="Reconstructed image")


def zoom_images(
    img: npt.NDArray, xlim: tuple[int, int], ylim: tuple[int, int],
    orig: Optional[npt.NDArray] = None, text: str = "Reconstructed"
    ):
    """
    Plots a grayscale image, `img`, and a portion of the image defined by `xlim` and `ylim`.

    Optionally compares `img` to `orig`.

    Note that `ylim[1]` should be **greater than** `ylim[0]` for a plot with the right side up.

    Args:
        img (npt.NDArray): image to plot
        xlim (tuple[int, int]): x-coordinates to start and end from
        ylim (tuple[int, int]): y-coordinates to end and start from
        orig (npt.NDArray): image to compare to `img`
        text (str): text to display on reconstructed image
    """
    if orig is not None:
        fig, axarr = plt.subplots(2, 2, figsize=(7, 6))

        axarr[0][0].imshow(orig, cmap="Greys")
        axarr[0][0].axis("off")
        axarr[0][0].set(title="Original")

        axarr[0][1].imshow(orig, cmap="Greys")
        axarr[0][1].axis("off")
        axarr[0][1].set_xlim(xlim)
        axarr[0][1].set_ylim(ylim)
        axarr[0][1].set(title="Zoomed In")

        axarr[1][0].imshow(img, cmap="Greys")
        axarr[1][0].axis("off")
        axarr[1][0].set(title=text)


        axarr[1][1].imshow(img, cmap="Greys")
        axarr[1][1].axis("off")
        axarr[1][1].set_xlim(xlim)
        axarr[1][1].set_ylim(ylim)
        axarr[1][1].set(title="Zoomed In")
    else:
        fig, axarr = plt.subplots(1, 2, figsize=(7, 3))

        axarr[0].imshow(img, cmap="Greys")
        axarr[0].axis("off")
        axarr[0].set(title="Original")

        axarr[1].imshow(img, cmap="Greys")
        axarr[1].axis("off")
        axarr[1].set_xlim(xlim)
        axarr[1].set_ylim(ylim)
        axarr[1].set(title="Zoomed In")

def svd_reconstruct(
    img: npt.NDArray, k: Optional[int] = None, cf: Optional[float] = None, epsilon: Optional[float] = None,
    store_vals: Optional[bool] = False,
    U: Optional[npt.NDArray] = None, sigma: Optional[npt.NDArray] = None, V: Optional[npt.NDArray] = None
    ) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    """
    Compress `img` and reconstruct it.

    The compression is defined either using `k` ranks, compression factor `cf`, or singular value threshold `epsilon`.

    Only one of `[k, cf, epsilon]` should be specified.

    Args:
        img (npt.NDArray): image to compress.
        k (int): rank of reconstructed matrix `A_ = U @ D @ V`.
        cf (float): compression factor, as amount of storage compared to original image.
        epsilon (float): threshold for singular values in `D`.
            Only components for which the corresponding singular value is larger than epsilon are used.
        store_vals (bool): whether to output U, sigma, V, for future use.
        U, sigma, V (npt.NDArray): results from previous run.

    Returns:
        npt.NDArray: reconstructed image.
        npt.NDArray, npt.NDArray, npt.NDArray: if store_vals, the calculated U, sigma, V
    """
    # https://stackoverflow.com/a/16801605
    args = iter([k, cf, epsilon])
    if not (any(args) and not any(args)):
        raise ValueError("Only one of (k, cf, epsilon) should be specified.")

    if cf:
        m, n = img.shape
        k = int(np.ceil(m * n * cf / (m + n + 1)))
    
    if sigma is None:
        U, sigma, V = np.linalg.svd(img)

    if epsilon:
        sigma = sigma[np.where(sigma > epsilon)]
        k = sigma.shape[0]

    # Reconstruct D
    D_ = np.zeros((k, k), dtype=float)
    D_[:k, :k] = np.diag(sigma[:k])

    # Limit rank to `k`
    U_ = U[:, :k]
    V_ = V[:k, :]

    # reconstruct
    if store_vals:
        return U_ @ D_ @ V_, U, sigma, V
    return U_ @ D_ @ V_

def svd_experiment(img: npt.NDArray):
    """
    Plots an experiment with number of components in svd compression.

    Args:
        img (npt.NDArray): image to experiment with.
    """
    xlim = (0, 340)
    ylim = (1390, 1060)
    sigma = None
    ks = [5, 10, 25, 50, 100, 150, 200]
    for k in ks:
        # attempt to speed up
        if sigma is None:
            img_recon, U, sigma, V = svd_reconstruct(img, k=k, store_vals=True)
        else:
            img_recon = svd_reconstruct(img, k=k, U=U, sigma=sigma, V=V)
        m, n = img.shape
        percent = 100 * k * (m + n + 1) / (m * n)
        zoom_images(
            img_recon, xlim, ylim, orig=img,
            text=f"k={k}, {percent:.2f}% storage"
            )