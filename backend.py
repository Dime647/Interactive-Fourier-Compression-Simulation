"""
backend.py — Numerical engine for 2-D Fourier image compression.

Provides:
  • Forward / inverse 2-D FFT with centered spectrum
  • Two compression strategies (Top-K magnitude, radial low-pass)
  • Fidelity metrics (MSE, PSNR, SSIM, retained energy)
  • Preset test-image loader
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.data import camera
from skimage.metrics import structural_similarity as ssim


# ──────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────

class CompressionMode(Enum):
    TOP_K = auto()
    RADIAL_CUTOFF = auto()


@dataclass(frozen=True)
class CompressionMetrics:
    """Quantitative telemetry for a single compression snapshot."""
    coefficients_zeroed_pct: float   # % of coefficients set to zero
    retained_energy_pct: float       # Parseval energy ratio (%)
    mse: float                       # Mean Squared Error
    psnr_db: float                   # Peak Signal-to-Noise Ratio (dB)
    ssim_index: float                # Structural Similarity Index


# ──────────────────────────────────────────────
# Preset image catalogue
# ──────────────────────────────────────────────

def _synthetic_shapes(size: int = 256) -> np.ndarray:
    """Generate a synthetic test pattern with circles, rectangles, and gradients."""
    img = np.zeros((size, size), dtype=np.float64)
    # Horizontal gradient band
    img[10:50, :] = np.linspace(0, 1, size)
    # Rectangle
    img[80:180, 60:200] = 0.9
    # Inner rectangle cutout
    img[100:160, 80:180] = 0.3
    # Circle
    yy, xx = np.ogrid[:size, :size]
    circle_mask = (xx - 192) ** 2 + (yy - 192) ** 2 <= 40 ** 2
    img[circle_mask] = 1.0
    # Small circle
    small = (xx - 64) ** 2 + (yy - 220) ** 2 <= 20 ** 2
    img[small] = 0.7
    # Diagonal line
    for i in range(size):
        if 0 <= i < size:
            img[i, min(i, size - 1)] = 1.0
    return img


def _checkerboard(size: int = 256, block: int = 32) -> np.ndarray:
    """Generate a checkerboard test pattern."""
    img = np.zeros((size, size), dtype=np.float64)
    for i in range(0, size, block):
        for j in range(0, size, block):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i:i + block, j:j + block] = 1.0
    return img


PRESET_IMAGES = {
    "Cameraman": lambda: camera().astype(np.float64) / 255.0,
    "Synthetic Shapes": lambda: _synthetic_shapes(),
    "Checkerboard": lambda: _checkerboard(),
}


def load_preset(name: str) -> np.ndarray:
    """Return a preset image normalised to [0, 1] float64."""
    return PRESET_IMAGES[name]()


def load_uploaded(file_bytes: bytes) -> np.ndarray:
    """Decode an uploaded image to [0, 1] grayscale float64."""
    pil = Image.open(io.BytesIO(file_bytes)).convert("L")
    arr = np.asarray(pil, dtype=np.float64) / 255.0
    # Limit size for interactive performance
    max_side = 512
    h, w = arr.shape
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return arr


# ──────────────────────────────────────────────
# 2-D FFT core
# ──────────────────────────────────────────────

def forward_fft(image: np.ndarray) -> np.ndarray:
    """Compute centred 2-D DFT of a grayscale image."""
    return np.fft.fftshift(np.fft.fft2(image))


def inverse_fft(spectrum: np.ndarray) -> np.ndarray:
    """Recover spatial image from centred spectrum; clamp to [0, 1]."""
    spatial = np.fft.ifft2(np.fft.ifftshift(spectrum))
    return np.clip(spatial.real, 0.0, 1.0)


def log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Compute log-scaled magnitude for visualisation."""
    return np.log1p(np.abs(spectrum))


# ──────────────────────────────────────────────
# Compression algorithms (fully vectorised)
# ──────────────────────────────────────────────

def compress_top_k(
    spectrum: np.ndarray, retain_pct: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dynamic magnitude thresholding — keep top *retain_pct* % of
    coefficients by absolute magnitude.

    Returns (masked_spectrum, boolean_mask).
    """
    magnitudes = np.abs(spectrum)
    total = magnitudes.size
    k = max(1, int(np.ceil(total * retain_pct / 100.0)))

    # Use argpartition for O(n) selection instead of full sort
    flat = magnitudes.ravel()
    threshold_idx = total - k
    partitioned = np.argpartition(flat, threshold_idx)
    threshold_val = flat[partitioned[threshold_idx]]

    mask = magnitudes >= threshold_val

    # If tie-breaking causes too many, trim from the weakest ties
    excess = mask.sum() - k
    if excess > 0:
        tie_mask = magnitudes == threshold_val
        tie_indices = np.argwhere(tie_mask)
        # Zero out `excess` of the tied positions
        for idx in tie_indices[:excess]:
            mask[tuple(idx)] = False

    masked = spectrum * mask
    return masked, mask


def compress_radial_cutoff(
    spectrum: np.ndarray, radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ideal low-pass zonal filter — zero all coefficients whose distance
    from the DC centre exceeds *radius* pixels.

    Returns (masked_spectrum, boolean_mask).
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist_sq <= radius ** 2
    masked = spectrum * mask
    return masked, mask


# ──────────────────────────────────────────────
# Fidelity metrics
# ──────────────────────────────────────────────

def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    full_spectrum: np.ndarray,
    mask: np.ndarray,
) -> CompressionMetrics:
    """Compute all quantitative telemetry in one pass."""
    total_coeffs = mask.size
    zeroed = int(np.count_nonzero(~mask))
    coefficients_zeroed_pct = 100.0 * zeroed / total_coeffs

    # Retained energy (Parseval)
    total_energy = np.sum(np.abs(full_spectrum) ** 2)
    kept_energy = np.sum(np.abs(full_spectrum * mask) ** 2)
    retained_energy_pct = (
        100.0 * kept_energy / total_energy if total_energy > 0 else 0.0
    )

    # MSE
    diff = original - reconstructed
    mse = float(np.mean(diff ** 2))

    # PSNR
    if mse < 1e-12:
        psnr_db = float("inf")
    else:
        psnr_db = 10.0 * np.log10(1.0 / mse)  # peak = 1.0

    # SSIM
    data_range = original.max() - original.min()
    if data_range < 1e-12:
        data_range = 1.0
    ssim_index = float(
        ssim(original, reconstructed, data_range=data_range)
    )

    return CompressionMetrics(
        coefficients_zeroed_pct=round(coefficients_zeroed_pct, 4),
        retained_energy_pct=round(retained_energy_pct, 4),
        mse=round(mse, 8),
        psnr_db=round(psnr_db, 2),
        ssim_index=round(ssim_index, 6),
    )


# ──────────────────────────────────────────────
# Visualisation helpers (Matplotlib → RGB array)
# ──────────────────────────────────────────────

def render_spectrum(data: np.ndarray, title: str = "") -> np.ndarray:
    """Render a log-magnitude spectrum to an RGB numpy array via Matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(data, cmap="gray", aspect="equal")
    ax.set_title(title, fontsize=10, color="white")
    ax.axis("off")
    fig.patch.set_facecolor("#0E1117")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[arg-type]
    rgb = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return rgb


def render_residual(
    original: np.ndarray, reconstructed: np.ndarray, gain: float = 5.0
) -> np.ndarray:
    """Render amplified |original − reconstructed| with 'inferno' colourmap."""
    diff = np.abs(original - reconstructed) * gain
    diff = np.clip(diff, 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=100)
    im = ax.imshow(diff, cmap="inferno", vmin=0, vmax=1, aspect="equal")
    ax.set_title("Residual Error (amplified ×{:.0f})".format(gain),
                 fontsize=10, color="white")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.patch.set_facecolor("#0E1117")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[arg-type]
    rgb = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return rgb
