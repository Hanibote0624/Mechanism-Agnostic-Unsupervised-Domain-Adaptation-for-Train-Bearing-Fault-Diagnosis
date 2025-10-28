
import numpy as np
from PIL import Image

def spectrogram_to_image(M_norm, width=640, height=480, invert_for_dark_high=True):
    """
    Convert a normalized spectrogram matrix (0..1) to a grayscale PIL image.
    By default, 'dark = high energy': intensity = 1 - M_norm.
    """
    M = np.asarray(M_norm, dtype=np.float32)
    M = np.clip(M, 0.0, 1.0)
    if invert_for_dark_high:
        M = 1.0 - M
    # scale to 0..255
    I = (M * 255.0 + 0.5).astype('uint8')
    # time axis -> x, freq -> y (origin at lower-left): flip vertically so low-freq at bottom
    I = np.flipud(I)
    # Resize to requested size with good quality
    img = Image.fromarray(I, mode="L").resize((width, height), resample=Image.BICUBIC)
    return img
