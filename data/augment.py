import cv2
from glob import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms as T
import albumentations as A
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

# The 12 corruption types used in RobustMedSAM (see paper Tables S2-S4).
# Modality-agnostic perturbations applied uniformly across all datasets:
#   gaussian_noise, gaussian_blur, contrast, brightness
# Modality-specific acquisition artifacts:
#   speckle_noise, salt_pepper_noise (ultrasound), rician_noise, rayleigh_noise,
#   step_motion (MRI), poisson_noise (CT/X-Ray/Microscopy), compression, color_jitter
DEGRADED_LIST = ['gauss_noise', 'gaussian_blur', 'contrast', 'brightness',
                 'compression', 'color_jitter', 'poisson_noise', 'speckle_noise',
                 'salt_pepper_noise', 'rician_noise', 'rayleigh_noise', 'step_motion']


def transform(case, path):
    ori_image = cv2.imread(path)
    clear_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    degraded_dict = {'gauss_noise': gauss_noise, 'gaussian_blur': gaussian_blur,
                     'contrast': contrast, 'brightness': brightness,
                     'compression': compression, 'color_jitter': color_jitter,
                     'poisson_noise': poisson_noise, 'speckle_noise': speckle_noise,
                     'salt_pepper_noise': salt_pepper_noise, 'rician_noise': rician_noise,
                     'rayleigh_noise': rayleigh_noise, 'step_motion': step_motion,
                     }

    image = degraded_dict[case](image)

    return image


# ----------------------------------------------------------------------------
# Modality-agnostic perturbations
# ----------------------------------------------------------------------------
def gauss_noise(image):
    # Additive Gaussian noise (var ~ 255, i.e. std ~ 16 on the [0, 255] scale).
    std = 16
    image = image.astype(np.float32)
    noise = np.random.normal(0, std, image.shape)
    augmented_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return augmented_image

def gaussian_blur(image):
    ksize = random.choice([7, 9, 11, 13])
    augmented_image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return augmented_image

def contrast(image):
    factor = random.uniform(1.6, 1.7)
    image = image.astype(np.float32)

    augmented_image = (image - 128) * factor + 128

    augmented_image = np.clip(augmented_image, 0, 255)
    augmented_image = augmented_image.astype(np.uint8)

    return augmented_image

def brightness(image):
    factor = random.uniform(0.6, 0.8)
    augmented_image = np.clip(image * factor, 0, 255).astype(np.uint8)

    return augmented_image


# ----------------------------------------------------------------------------
# Modality-specific acquisition artifacts
# ----------------------------------------------------------------------------
def compression(image):
    # Strong JPEG compression artifacts (quality in [5, 10]).
    quality = random.randint(5, 10)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode('.jpg', bgr, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    augmented_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    return augmented_image

def color_jitter(image):
    brightness = 0.7
    contrast = 0.3
    saturation = 0.7
    hue = 0.7

    transform = A.Compose(
        [A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=1)],
    )

    image = transform(image=image)
    image = image['image']

    return image

def poisson_noise(image):
    # Shot noise: a smaller scale c yields stronger Poisson noise.
    c = 25
    image = image.astype(np.float32) / 255.0
    augmented_image = np.random.poisson(image * c) / float(c)
    augmented_image = np.clip(augmented_image, 0, 1) * 255
    augmented_image = augmented_image.astype(np.uint8)

    return augmented_image

def speckle_noise(image):
    # Multiplicative Gaussian noise (characteristic of ultrasound / OCT).
    var = 0.25
    image = image.astype(np.float32)
    noise = np.random.normal(0, var ** 0.5, image.shape)
    augmented_image = image + image * noise
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    return augmented_image

def salt_pepper_noise(image):
    amount = 0.1
    s_vs_p = 0.5
    augmented_image = image.copy()
    h, w = image.shape[:2]

    num_salt = int(amount * h * w * s_vs_p)
    ys = np.random.randint(0, h, num_salt)
    xs = np.random.randint(0, w, num_salt)
    augmented_image[ys, xs] = 255

    num_pepper = int(amount * h * w * (1.0 - s_vs_p))
    ys = np.random.randint(0, h, num_pepper)
    xs = np.random.randint(0, w, num_pepper)
    augmented_image[ys, xs] = 0

    return augmented_image

def rician_noise(image):
    # Rician noise on the magnitude signal (characteristic of MRI):
    # out = sqrt((I + n1)^2 + n2^2),  n1, n2 ~ N(0, sigma).
    sigma = 25
    image = image.astype(np.float32)
    n1 = np.random.normal(0, sigma, image.shape)
    n2 = np.random.normal(0, sigma, image.shape)
    augmented_image = np.sqrt((image + n1) ** 2 + n2 ** 2)
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    return augmented_image

def rayleigh_noise(image):
    # Additive Rayleigh-distributed noise, mean-centered to preserve brightness.
    scale = 40
    image = image.astype(np.float32)
    noise = np.random.rayleigh(scale, image.shape)
    augmented_image = image + noise - scale * np.sqrt(np.pi / 2.0)
    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    return augmented_image

def step_motion(image):
    # Step-wise rigid motion artifact (MRI). A subset of phase-encode lines in
    # k-space is corrupted with a linear phase ramp, producing ghosting.
    shift = 5
    image = image.astype(np.float32)
    augmented_image = np.zeros_like(image)

    for c in range(image.shape[2]):
        f = np.fft.fftshift(np.fft.fft2(image[:, :, c]))
        rows, cols = f.shape
        ramp = np.exp(-2j * np.pi * shift * np.arange(cols) / cols)
        # Corrupt the lower half of the phase-encode lines (a discrete motion step).
        f[rows // 2:, :] = f[rows // 2:, :] * ramp[np.newaxis, :]
        rec = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
        augmented_image[:, :, c] = rec

    augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)

    return augmented_image


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="all_data/train")
parser.add_argument("--case", type=str, default=None)

opt = parser.parse_args()

image_path = os.path.join(opt.data_dir, 'clear')
image_list = sorted(glob(os.path.join(image_path, '*jpg')))
print('Number of images: {}'.format(len(image_list)))

if opt.case is None:
    degraded_list = DEGRADED_LIST

else:
    degraded_list = [opt.case]
    print('Processing {} only!'.format(opt.case))


for j, degraded_type in enumerate(degraded_list):
    case = degraded_type
    save_folder = os.path.join(opt.data_dir, case)
    os.makedirs(save_folder, exist_ok=True)
    print('Generating {} images. Degraded images will be saved to {} ...'.format(degraded_type, save_folder))

    for i, path in enumerate(tqdm(image_list)):
        augmented_image = transform(case, path)
        fname = path.split('/')[-1]
        save_path = os.path.join(save_folder, fname)
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, augmented_image)
