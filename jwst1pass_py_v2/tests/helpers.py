"""Shared fixtures for jwst1pass tests."""

import numpy as np
import pytest
import sys, os
from scipy.ndimage import spline_filter

# Add package root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jwst1pass.core import _eval_psf_grad_fast


PSF_SCALE = 4
PSF_SIZE  = 101


def make_gaussian_psf(fwhm_pix=2.0, psf_scale=PSF_SCALE, size=PSF_SIZE):
    """Supersampled Gaussian PSF normalised so sum/psf_scale² ≈ 1."""
    sigma_ss = fwhm_pix * psf_scale / 2.3548
    c = size // 2
    y, x = np.mgrid[:size, :size]
    r2 = (x - c) ** 2 + (y - c) ** 2
    psf = np.exp(-0.5 * r2 / sigma_ss**2)
    psf /= psf.sum() / psf_scale**2
    return psf


def inject_stars(stars, psf, psf_scale, sky=10.0, image_size=(256, 256),
                 gain=1.0, read_noise=10.0, seed=42):
    """Return noisy synthetic image with stars injected at known (x, y, flux).

    stars : list of (x, y, flux)
    """
    ny, nx = image_size
    image = np.full((ny, nx), sky, dtype=np.float64)
    hw = 15  # large injection window
    
    psf_coeffs = spline_filter(psf, order=3, output=np.float64)

    for x0, y0, flux in stars:
        xi = int(round(x0))
        yi = int(round(y0))
        dx = x0 - xi
        dy = y0 - yi
        y_lo = max(0, yi - hw);  y_hi = min(ny, yi + hw + 1)
        x_lo = max(0, xi - hw);  x_hi = min(nx, xi + hw + 1)
        diy = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
        dix = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
        P, _, _ = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix, diy, psf_scale)
        image[y_lo:y_hi, x_lo:x_hi] += flux * P

    rng = np.random.default_rng(seed)
    # Poisson noise (on source + sky)
    image_pos = np.maximum(image, 0.0)
    noisy = rng.poisson(image_pos).astype(np.float64)
    # Read noise
    noisy += rng.normal(0.0, read_noise, size=(ny, nx))
    return noisy
