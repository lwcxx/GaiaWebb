"""Tests for multi-pass photometry and neighbour subtraction in jwst1pass."""

import numpy as np
import pytest
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE

from jwst1pass.multipass import run_photometry, subtract_stars


# ---------------------------------------------------------------------------
# 1. Multi-pass completeness
# ---------------------------------------------------------------------------

def test_multipass_completeness(gauss_psf, psf_cube, psf_positions):
    """Pass 2 should detect a faint neighbour missed or poorly measured in pass 1."""
    sky = 10.0
    bright = (80.0, 80.0, 8000.0)
    faint  = (85.0, 80.0, 2000.0)   # 5 pixels away, 4× fainter
    image = inject_stars([bright, faint], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(160, 160), gain=1.0, read_noise=10.0)

    xs, ys = psf_positions
    records_2pass, residual = run_photometry(
        data=image, psf_cube=psf_cube, xs=xs, ys=ys,
        psf_scale=PSF_SCALE, hw=5, hmin=3, fmin=100.0,
        n_passes=2, gain=1.0, read_noise=10.0, zero_point=25.0,
    )

    # After 2 passes we should have both stars
    assert len(records_2pass) >= 2, \
        f"Only {len(records_2pass)} stars found in 2-pass run, expected ≥ 2"

    xs_fit = np.array([r.x for r in records_2pass])
    ys_fit = np.array([r.y for r in records_2pass])

    for x0, y0, _ in [bright, faint]:
        dists = np.hypot(xs_fit - x0, ys_fit - y0)
        assert dists.min() < 1.5, \
            f"No star within 1.5 px of injected position ({x0}, {y0})"


# ---------------------------------------------------------------------------
# 2. Neighbour subtraction
# ---------------------------------------------------------------------------

def test_neighbour_subtraction(gauss_psf, psf_cube, psf_positions):
    """After subtracting a bright star, residual at its position should be small."""
    sky = 10.0
    x0, y0, flux = 64.0, 64.0, 20000.0
    image = inject_stars([(x0, y0, flux)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=1.0, read_noise=10.0)

    xs, ys = psf_positions
    records, _ = run_photometry(
        data=image, psf_cube=psf_cube, xs=xs, ys=ys,
        psf_scale=PSF_SCALE, hw=5, hmin=3, fmin=100.0, n_passes=1,
    )

    resid_img = image.copy()
    subtract_stars(resid_img, records, psf_cube, xs, ys, PSF_SCALE, hw=5)

    # Peak residual at star position should be low
    peak_residual = abs(resid_img[int(round(y0)), int(round(x0))] - sky)
    # 10.0 read noise + Poisson noise on sky
    expected_noise = np.sqrt(10.0**2 + sky)
    assert peak_residual < 5.0 * expected_noise, \
        f"Residual peak {peak_residual:.2f} too large"
