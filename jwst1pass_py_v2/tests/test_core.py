"""Tests for PSF evaluation, source finding, and fitting in jwst1pass."""

import numpy as np
import pytest
from scipy.ndimage import spline_filter
from helpers import make_gaussian_psf, inject_stars, PSF_SCALE, PSF_SIZE

from jwst1pass.core import (
    _eval_psf_grad_fast, detect_stars, fit_star, StarRecord,
)
from jwst1pass.diagnostics import summarize_catalog


# ---------------------------------------------------------------------------
# 1. PSF normalisation
# ---------------------------------------------------------------------------

def test_psf_normalisation(gauss_psf):
    """sum(P over window) should be ≈ 1 for a centred star."""
    psf_scale = PSF_SCALE
    hw = 5
    diy = np.arange(-hw, hw + 1)[:, np.newaxis]
    dix = np.arange(-hw, hw + 1)[np.newaxis, :]
    
    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    P, _, _ = _eval_psf_grad_fast(psf_coeffs, 0.0, 0.0, dix, diy, psf_scale)
    
    assert abs(P.sum() - 1.0) < 0.05, f"PSF window sum = {P.sum():.4f}, expected ≈ 1"


# ---------------------------------------------------------------------------
# 2. Gradient correctness
# ---------------------------------------------------------------------------

def test_gradient_correctness(gauss_psf):
    """Analytical dP/dx from _eval_psf_grad_fast must agree with numerical FD."""
    psf_scale = PSF_SCALE
    hw = 5
    diy = np.arange(-hw, hw + 1)[:, np.newaxis]
    dix = np.arange(-hw, hw + 1)[np.newaxis, :]

    dx, dy = 0.2, -0.1
    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    _, dPdx_analytic, dPdy_analytic = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix, diy, psf_scale)

    eps = 1e-4  # detector pixels
    P_xp, _, _ = _eval_psf_grad_fast(psf_coeffs, dx + eps, dy, dix, diy, psf_scale)
    P_xm, _, _ = _eval_psf_grad_fast(psf_coeffs, dx - eps, dy, dix, diy, psf_scale)
    dPdx_num = (P_xp - P_xm) / (2 * eps)

    P_yp, _, _ = _eval_psf_grad_fast(psf_coeffs, dx, dy + eps, dix, diy, psf_scale)
    P_ym, _, _ = _eval_psf_grad_fast(psf_coeffs, dx, dy - eps, dix, diy, psf_scale)
    dPdy_num = (P_yp - P_ym) / (2 * eps)

    for grad_a, grad_n, label in [
        (dPdx_analytic, dPdx_num, 'dP/dx'),
        (dPdy_analytic, dPdy_num, 'dP/dy'),
    ]:
        sig = np.abs(grad_n) > 0.10 * np.abs(grad_n).max()
        if sig.any():
            rel_err = np.abs(grad_a[sig] - grad_n[sig]) / (np.abs(grad_n[sig]) + 1e-30)
            assert rel_err.max() < 0.10, f"max {label} rel error = {rel_err.max():.4f}"


# ---------------------------------------------------------------------------
# 3. Position recovery
# ---------------------------------------------------------------------------

def test_position_recovery(gauss_psf):
    """Recovered position should be within 0.05 px of injected position."""
    x_true, y_true, flux_true = 127.37, 63.82, 5000.0
    sky = 10.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 256), gain=1.0, read_noise=10.0)

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    rec = fit_star(
        data=image, x0=round(x_true), y0=round(y_true),
        psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
        hw=5, sky_init=sky, gain=1.0, read_noise=10.0,
        max_iter=15, tol=1e-5, zero_point=0.0, pass_number=1,
    )

    assert abs(rec.x - x_true) < 0.05, f"x error = {abs(rec.x - x_true):.4f} px"
    assert abs(rec.y - y_true) < 0.05, f"y error = {abs(rec.y - y_true):.4f} px"


# ---------------------------------------------------------------------------
# 4. Flux recovery
# ---------------------------------------------------------------------------

def test_flux_recovery(gauss_psf):
    """Recovered flux should be within 5% of injected flux for S/N > 20."""
    x_true, y_true, flux_true = 100.0, 100.0, 10000.0
    sky = 10.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(200, 200), gain=1.0, read_noise=10.0)

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
        hw=5, sky_init=sky, gain=1.0, read_noise=10.0,
        max_iter=15, tol=1e-5, zero_point=0.0, pass_number=1,
    )

    frac_err = abs(rec.flux - flux_true) / flux_true
    assert frac_err < 0.05, f"flux fractional error = {frac_err:.4f}"


# ---------------------------------------------------------------------------
# 5. Covariance positive-definite
# ---------------------------------------------------------------------------

def test_covariance_positive_definite(gauss_psf):
    x_true, y_true, flux_true = 80.5, 80.5, 5000.0
    sky = 10.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(160, 160), gain=1.0, read_noise=10.0)

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
        hw=5, sky_init=sky, gain=1.0, read_noise=10.0,
        max_iter=15, tol=1e-5, zero_point=0.0, pass_number=1,
    )

    eigvals = np.linalg.eigvalsh(rec.cov)
    assert np.all(eigvals > 0), f"Covariance not positive definite: eigvals={eigvals}"


# ---------------------------------------------------------------------------
# 6. flux_err consistency with expected S/N
# ---------------------------------------------------------------------------

def test_flux_err_consistency(gauss_psf):
    """S/N from covariance should agree with Poisson S/N to within factor 2."""
    flux_true = 10000.0
    sky = 10.0
    x_true, y_true = 64.0, 64.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=1.0, read_noise=10.0)

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
        hw=5, sky_init=sky, gain=1.0, read_noise=10.0,
        max_iter=15, tol=1e-5, zero_point=0.0, pass_number=1,
    )
    
    # Fill errors (normally done by apply_chi2_inflation)
    rec.flux_err = np.sqrt(rec.cov[0, 0])

    sn_fit = rec.flux / rec.flux_err
    sn_poisson = np.sqrt(flux_true)  # simplified: photon noise on source
    assert 0.5 * sn_poisson < sn_fit < 2.0 * sn_poisson, \
        f"S/N mismatch: fit={sn_fit:.1f}, poisson={sn_poisson:.1f}"


# ---------------------------------------------------------------------------
# 7. Edge pixels
# ---------------------------------------------------------------------------

def test_edge_pixels(gauss_psf):
    """Stars within hw pixels of the image boundary must not raise exceptions."""
    image = inject_stars([(1.5, 1.5, 3000.0)], gauss_psf, PSF_SCALE,
                         sky=10.0, image_size=(64, 64), gain=1.0, read_noise=10.0)

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    rec = fit_star(
        data=image, x0=1.5, y0=1.5,
        psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
        hw=5, sky_init=10.0, gain=1.0, read_noise=10.0,
        max_iter=5, tol=1e-4, zero_point=0.0, pass_number=1,
    )
    assert isinstance(rec, StarRecord)


# ---------------------------------------------------------------------------
# 8. Source detection
# ---------------------------------------------------------------------------

def test_detect_stars(gauss_psf):
    """A bright injected star should be detected by detect_stars."""
    x_true, y_true, flux_true = 64.0, 64.0, 5000.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=10.0, image_size=(128, 128), gain=1.0, read_noise=10.0)

    stars = detect_stars(image, hmin=3, fmin=100.0)
    assert len(stars) >= 1
    # Brightest should be near injected position
    best = sorted(stars, key=lambda s: s['flux'], reverse=True)[0]
    assert abs(best['x'] - x_true) < 1.5
    assert abs(best['y'] - y_true) < 1.5


# ---------------------------------------------------------------------------
# 9. Sigma clipping handles cosmic rays
# ---------------------------------------------------------------------------

def test_sigma_clip_cosmic_ray(gauss_psf):
    """Sigma clipping should improve position accuracy when a cosmic ray is present."""
    x_true, y_true, flux_true = 64.0, 64.0, 8000.0
    sky = 10.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=1.0, read_noise=10.0)
    # Inject a bright cosmic ray 2 pixels from centre
    image[int(y_true), int(x_true) + 2] += 50000.0

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    kw = dict(psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
              hw=5, sky_init=sky, gain=1.0, read_noise=10.0,
              max_iter=15, tol=1e-5, zero_point=0.0, pass_number=1)

    rec_no  = fit_star(data=image, x0=x_true, y0=y_true, sigma_clip=False, **kw)
    rec_yes = fit_star(data=image, x0=x_true, y0=y_true, sigma_clip=True,  **kw)

    err_no  = abs(rec_no.x  - x_true)
    err_yes = abs(rec_yes.x - x_true)
    assert err_yes <= err_no + 0.02, \
        f"Sigma clip made position worse: no_clip={err_no:.4f}px, clip={err_yes:.4f}px"
    assert err_yes < 0.3, f"Clipped position error too large: {err_yes:.4f}px"


# ---------------------------------------------------------------------------
# 10. summarize_catalog
# ---------------------------------------------------------------------------

def test_summarize_catalog(gauss_psf):
    """summarize_catalog should return sane statistics without crashing."""
    x_true, y_true, flux_true = 64.0, 64.0, 5000.0
    sky = 10.0
    image = inject_stars([(x_true, y_true, flux_true)], gauss_psf, PSF_SCALE,
                         sky=sky, image_size=(128, 128), gain=1.0, read_noise=10.0)

    psf_coeffs = spline_filter(gauss_psf, order=3, output=np.float64)
    rec = fit_star(
        data=image, x0=x_true, y0=y_true,
        psf_coeffs=psf_coeffs, psf_scale=PSF_SCALE,
        hw=5, sky_init=sky, gain=1.0, read_noise=10.0,
        max_iter=15, tol=1e-5, zero_point=0.0, pass_number=1,
    )
    rec.n_neighbors = 0; rec.dist_nearest = np.inf; rec.dist_nearest_brighter = np.inf

    stats = summarize_catalog([rec], verbose=False)
    assert stats['n_stars'] == 1
    assert np.isfinite(stats['qfit_median'])
    assert np.isfinite(stats['chi2_median'])
    assert stats['snr_median'] > 0
