"""Image corruption and robustness benchmarking utilities.

This module provides common corruption types (noise, blur, weather, digital)
used to evaluate the out-of-distribution robustness of deep learning models.
"""

from io import BytesIO
import os
import cv2
import numpy as np
from PIL import Image, Image as PILImage
from pkg_resources import resource_filename
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import skimage as sk
from skimage.filters import gaussian
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100, mnist
from tensorflow.keras.utils import to_categorical
from wand.api import library as wandlibrary
from wand.image import Image as WandImage


# Global dataset normalization statistics
(train_images2, train_labels), (test_images2, test_labels) = cifar100.load_data()
mean = np.mean(train_images2, axis=(0, 1, 2, 3))
std = np.std(train_images2, axis=(0, 1, 2, 3))
test_labels = to_categorical(test_labels, 100)


class MotionImage(WandImage):
    """Wand Image wrapper supporting motion blur operations via C-API binding."""

    def motion_blur(self, radius: float = 0.0, sigma: float = 0.0, angle: float = 0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def clipped_zoom(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Zooms into an image array and clips borders to retain original size."""
    h = img.shape[1]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(img[:, top : top + ch, top : top + ch], (1, zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[1] - h) // 2
    return img[:, trim_top : trim_top + h, trim_top : trim_top + h]


def gaussian_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies Gaussian noise distortion."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies Poisson shot noise distortion."""
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x)
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255


def zoom_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies radial zoom blur."""
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def pixelate(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Downsamples and upsamples image to create pixelation effects."""
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    ims = []
    for d in range(x.shape[0]):
        im = PIL.Image.fromarray(x[d])
        im = im.resize((int(32 * c), int(32 * c)), Image.BOX)
        im = im.resize((32, 32), Image.BOX)
        ims.append(im)
    return np.stack(ims, 0)


def contrast(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Modifies image contrast relative to mean image luminance."""
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Adjusts image brightness in HSV space."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, :, 2] = np.clip(x[:, :, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def saturate(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Adjusts color saturation channel in HSV space."""
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x)
    x = sk.color.rgb2hsv(x)
    x[:, :, :, 1] = np.clip(x[:, :, :, 1] * c[0] + c[1], -1.8816435, 2.0934134)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1)


def impulse_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies Salt-and-Pepper impulse noise."""
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(image_batch: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies JPEG compression artifacts to an image batch (N, H, W, C)."""
    c = [25, 18, 15, 10, 7][severity - 1]
    processed_images = []

    for i in range(image_batch.shape[0]):
        x = image_batch[i]
        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)

        pil_img = Image.fromarray(x)
        output = BytesIO()
        pil_img.save(output, "JPEG", quality=c)
        output.seek(0)

        compressed_img = Image.open(output)
        processed_images.append(np.array(compressed_img))

    return np.array(processed_images)


def glass_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies Gaussian blur followed by local pixel shuffling."""
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) * 255)

    for i in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1) * 255


def fourier_style_mix(img1: tf.Tensor, img2: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    """Mixes Fourier amplitude of img2 into img1 while preserving phase structure of img1."""
    fft1 = tf.signal.rfft2d(tf.transpose(img1, [0, 3, 1, 2]))
    fft2 = tf.signal.rfft2d(tf.transpose(img2, [0, 3, 1, 2]))

    amp1, phase1 = tf.abs(fft1), tf.math.angle(fft1)
    amp2 = tf.abs(fft2)

    mixed_amp = (1 - alpha) * amp1 + alpha * amp2
    mixed_fft = tf.cast(mixed_amp, tf.complex64) * tf.exp(
        tf.complex(tf.zeros_like(phase1), phase1)
    )

    result = tf.signal.irfft2d(mixed_fft)
    result = tf.transpose(result, [0, 2, 3, 1])
    return tf.clip_by_value(result, 0.0, 1.0)


def snow(images: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies simulated snow weather conditions."""
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
    ][severity - 1]

    N, H, W, C = images.shape
    corrupted_images = np.empty_like(images, dtype=np.float32)

    for i in range(N):
        x = images[i].astype(np.float32) / 255.0
        snow_layer = np.random.normal(size=(H, W), loc=c[0], scale=c[1])
        snow_input = snow_layer[np.newaxis, ..., np.newaxis]

        try:
            snow_layer = clipped_zoom(snow_input, (1, c[2], c[2], 1))
            snow_layer = snow_layer.squeeze()
        except Exception:
            zoom_h, zoom_w = int(H * c[2]), int(W * c[2])
            snow_layer = cv2.resize(snow_layer, (zoom_w, zoom_h))
            sh, sw = snow_layer.shape
            ymin = max(0, (sh - H) // 2)
            xmin = max(0, (sw - W) // 2)
            snow_layer = snow_layer[ymin : ymin + H, xmin : xmin + W]
            snow_layer = cv2.resize(snow_layer, (W, H))

        if snow_layer.shape[:2] != (H, W):
            snow_layer = cv2.resize(snow_layer, (W, H))

        if len(snow_layer.shape) == 2:
            snow_layer = snow_layer[..., np.newaxis]

        snow_layer[snow_layer < c[3]] = 0

        snow_img_pil = PILImage.fromarray(
            (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
        )
        output = BytesIO()
        snow_img_pil.save(output, format="PNG")

        snow_motion = MotionImage(blob=output.getvalue())
        snow_motion.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

        snow_layer = cv2.imdecode(
            np.frombuffer(snow_motion.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )

        if snow_layer.shape[:2] != (H, W):
            snow_layer = cv2.resize(snow_layer, (W, H))
        snow_layer = snow_layer.astype(np.float32) / 255.0
        if len(snow_layer.shape) == 2:
            snow_layer = snow_layer[..., np.newaxis]

        gray_x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(H, W, 1)
        x = c[6] * x + (1 - c[6]) * np.maximum(x, gray_x * 1.5 + 0.5)

        corrupted = np.clip(x + snow_layer + np.rot90(snow_layer, k=2, axes=(0, 1)), 0, 1) * 255.0
        corrupted_images[i] = corrupted

    return corrupted_images.astype(images.dtype)


def frost(images: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies frost texture overlays to an image batch."""
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]

    N, H, W, C = images.shape
    corrupted_images = np.empty_like(images, dtype=np.float32)

    filenames = ["frost/frost1.png", "frost/frost2.png", "frost/frost3.png", "frost/frost4.jpg", "frost/frost5.jpg", "frost/frost6.jpg"]

    for i in range(N):
        idx = np.random.randint(len(filenames))
        filename = filenames[idx]

        if not os.path.exists(filename):
            alt_filename = filename.replace(".jpg", ".png")
            if os.path.exists(alt_filename):
                filename = alt_filename
            else:
                raise FileNotFoundError(f"Could not find frost image file: {filename}")

        frost_img = cv2.imread(filename)
        if frost_img is None:
            raise FileNotFoundError(f"Failed to read image: {filename}")

        fh, fw = frost_img.shape[:2]

        if fh < H or fw < W:
            frost_img = cv2.resize(frost_img, (max(fw, W), max(fh, H)))
            fh, fw = frost_img.shape[:2]

        x_start = np.random.randint(0, fh - H + 1)
        y_start = np.random.randint(0, fw - W + 1)
        frost_crop = frost_img[x_start : x_start + H, y_start : y_start + W][..., [2, 1, 0]]

        img = images[i].astype(np.float32)
        corrupted = np.clip(c[0] * img + c[1] * frost_crop, 0, 255)
        corrupted_images[i] = corrupted

    return corrupted_images.astype(images.dtype)


def motion_blur(images: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies trajectory motion blur to an image batch."""
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    if isinstance(images, list):
        images = np.array(images)

    corrupted_images = np.empty_like(images, dtype=np.uint8)

    for i in range(len(images)):
        img_np = images[i].astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)

        output = BytesIO()
        pil_img.save(output, format="PNG")

        x = MotionImage(blob=output.getvalue())
        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        decoded = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

        if len(decoded.shape) == 3 and decoded.shape[2] >= 3:
            processed = decoded[..., [2, 1, 0]]
        else:
            processed = np.array([decoded, decoded, decoded]).transpose((1, 2, 0))

        corrupted_images[i] = np.clip(processed, 0, 255).astype(np.uint8)

    return corrupted_images


def elastic_transform(images: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies elastic grid deformations."""
    H, W = images.shape[1], images.shape[2]
    scale = W / 244.0

    c_list = [
        (244 * 2, 244 * 0.7, 244 * 0.1),
        (244 * 2, 244 * 0.08, 244 * 0.2),
        (244 * 0.05, 244 * 0.01, 244 * 0.02),
        (244 * 0.07, 244 * 0.01, 244 * 0.02),
        (244 * 0.12, 244 * 0.01, 244 * 0.02),
    ][severity - 1]

    c = (c_list[0] * scale, c_list[1] * scale, c_list[2] * scale)
    corrupted_images = np.empty_like(images, dtype=np.float32)

    for i in range(len(images)):
        image = np.array(images[i], dtype=np.float32) / 255.0
        shape = image.shape
        shape_size = shape[:2]

        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (
            gaussian(
                np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
            )
            * c[0]
        ).astype(np.float32)
        dy = (
            gaussian(
                np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
            )
            * c[0]
        ).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )

        corrupted_image = (
            np.clip(map_coordinates(image, indices, order=1, mode="reflect").reshape(shape), 0, 1)
            * 255
        )
        corrupted_images[i] = corrupted_image

    return corrupted_images.astype(images.dtype)


def augment_images(x: tf.Tensor, x_donor: tf.Tensor) -> tf.Tensor:
    """Applies composite dynamic data augmentations including Fourier style mixing."""
    x = tf.clip_by_value(x * tf.random.uniform([], 0.6, 1.4), 0, 1)
    x = tf.clip_by_value(x * tf.random.uniform([1, 1, 3], 0.7, 1.3), 0, 1)
    x = tf.pow(tf.clip_by_value(x, 1e-8, 1.0), tf.random.uniform([], 0.6, 1.8))
    x = fourier_style_mix(x, x_donor, alpha=tf.random.uniform([], 0.1, 0.5))
    x = tf.keras.layers.RandomRotation(0.15)(x)
    return x


def plasma_fractal(mapsize: int = 256, wibbledecay: int = 3) -> np.ndarray:
    """Generates a fractal heightmap using the diamond-square algorithm."""
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies plasma fractal fog corruption."""
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:32, :32][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def disk(radius: float, alias_blur: float = 0.1, dtype=np.float32) -> np.ndarray:
    """Generates an anti-aliased disk kernel for defocus blur."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    """Applies defocus blur using disk kernels."""
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])
    ims = []

    for i in range(x.shape[0]):
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[i, :, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))
        ims.append(np.clip(channels, 0, 1) * 255)
    return np.stack(ims, 0)


def test_robustness(model: tf.keras.Model, images: np.ndarray) -> list[float]:
    """Evaluates a model across 15 corruptions (5 severities each) and returns mean error rates."""
    errs = []
    nums = 5

    print("jpeg")
    add = 0
    for c in range(nums):
        images_n = jpeg_compression(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("motion")
    add = 0
    for c in range(nums):
        images_n = motion_blur(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("elastic")
    add = 0
    for c in range(nums):
        images_n = elastic_transform(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("snow")
    add = 0
    for c in range(nums):
        images_n = snow(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("frost")
    add = 0
    for c in range(nums):
        images_n = frost(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("glass_blur")
    add = 0
    for c in range(5):
        corrupted_images = glass_blur(images, severity=c + 1)
        images_n = np.array(corrupted_images)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("contrast")
    add = 0
    for c in range(5):
        images_n = contrast(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("brightness")
    add = 0
    for c in range(5):
        images_n = brightness(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("shot")
    add = 0
    for c in range(5):
        images_n = shot_noise(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("impulse")
    add = 0
    for c in range(5):
        images_n = impulse_noise(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("gaussian")
    add = 0
    for c in range(5):
        images_n = gaussian_noise(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("defocus")
    add = 0
    for c in range(5):
        images_n = defocus_blur(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("zoom")
    add = 0
    for c in range(5):
        images_n = zoom_blur(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("pixelate")
    add = 0
    for c in range(5):
        images_n = pixelate(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    print("fog")
    add = 0
    for c in range(5):
        images_n = fog(images, severity=c + 1)
        images_n = (images_n - mean) / (std + 1e-7)
        l, sc = model.evaluate(images_n, test_labels)
        add += 100 * sc
    print((500 - add) / 5)
    errs.append((500 - add) / 5)

    return errs