#!/usr/bin/env python
"""Assess PSF interpolation accuracy and its propagation to position uncertainty for jwst1pass.

Five tests:
  1. Numba vs scipy B-spline evaluation consistency
  2. FD gradient accuracy vs 100x finer FD step (as reference "truth")
  3. PSF spatial cache error (checks interpolation across the spatial grid)
  4. Systematic position bias — noiseless injection, isolates model errors
  5. Noisy MC position precision — confirms statistical floor
"""

import sys, os
# Add package root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.ndimage import spline_filter, map_coordinates

from jwst1pass.core import (
    _eval_psf_grad_fast, _NUMBA, fit_star,
)
from jwst1pass.io import load_psf_cube
from jwst1pass.utils import interpolate_psf

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
HW    = 5       # fit window half-width (typical for JWST)
SKY   = 10.0
GAIN  = 1.0
RN    = 10.0

STDPSF_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'lib', 'STDPSFs', 'MIRI', 'STDPSF_MIRI_F770W.fits')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gaussian_psf(fwhm=2.0, scale=4, size=101):
    sigma_ss = fwhm * scale / 2.3548
    c = size // 2
    y, x = np.mgrid[:size, :size]
    r2 = (x - c)**2 + (y - c)**2
    psf = np.exp(-0.5 * r2 / sigma_ss**2)
    psf /= psf.sum() / scale**2
    return psf


def load_real_psf():
    """Load the real STDPSF; return (cube, xs, ys, psf_scale, center_psf)."""
    cube, xs, ys, psf_scale, _ = load_psf_cube(STDPSF_PATH)
    # Pick the centre-of-field PSF model for single-PSF tests
    mid_idx = len(cube) // 2
    center_psf = cube[mid_idx].copy()
    return cube, xs, ys, psf_scale, center_psf


def inject_noiseless(psf_coeffs, x0, y0, flux, psf_scale, sky=SKY, image_size=(256, 256)):
    ny, nx = image_size
    image  = np.full((ny, nx), sky, dtype=np.float64)
    xi = int(round(x0)); yi = int(round(y0))
    dx = x0 - xi;        dy = y0 - yi
    hw_inj = 15
    y_lo = max(0, yi - hw_inj); y_hi = min(ny, yi + hw_inj + 1)
    x_lo = max(0, xi - hw_inj); x_hi = min(nx, xi + hw_inj + 1)
    diy2d = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
    dix2d = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
    P, _, _ = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix2d, diy2d, psf_scale)
    image[y_lo:y_hi, x_lo:x_hi] += flux * P
    return image


def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ---------------------------------------------------------------------------
# 1. Numba vs scipy B-spline consistency
# ---------------------------------------------------------------------------

def test_numba_scipy(psf, psf_scale, label, n_positions=500, rng=None):
    if not _NUMBA:
        print(f"  [{label}] Numba NOT available — skipping")
        return
    if rng is None:
        rng = np.random.default_rng(0)

    coeffs = spline_filter(psf, order=3, output=np.float64)
    size = psf.shape[0]
    half_ss = size // 2

    dix_1d = np.arange(-HW, HW + 1)
    diy_1d = np.arange(-HW, HW + 1)
    diy, dix = np.meshgrid(diy_1d, dix_1d, indexing='ij')

    dxs = rng.uniform(-0.5, 0.5, n_positions)
    dys = rng.uniform(-0.5, 0.5, n_positions)

    max_P  = max_Gx = max_Gy = 0.0

    from jwst1pass.core import _nb_eval_psf_grad
    for dx, dy in zip(dxs, dys):
        xs_f = (half_ss + (dix.ravel() - dx) * psf_scale).astype(np.float64)
        ys_f = (half_ss + (diy.ravel() - dy) * psf_scale).astype(np.float64)
        kw = dict(order=3, mode='nearest', prefilter=False)

        P_sc    = map_coordinates(coeffs, [ys_f, xs_f    ], **kw)
        Pxp_sc  = map_coordinates(coeffs, [ys_f, xs_f + 1], **kw)
        Pxm_sc  = map_coordinates(coeffs, [ys_f, xs_f - 1], **kw)
        Pyp_sc  = map_coordinates(coeffs, [ys_f + 1, xs_f], **kw)
        Pym_sc  = map_coordinates(coeffs, [ys_f - 1, xs_f], **kw)
        Gx_sc = (Pxm_sc - Pxp_sc) * 0.5 * psf_scale
        Gy_sc = (Pym_sc - Pyp_sc) * 0.5 * psf_scale

        n = len(xs_f)
        P_nb  = np.empty(n); Gx_nb = np.empty(n); Gy_nb = np.empty(n)
        _nb_eval_psf_grad(coeffs, ys_f, xs_f, float(psf_scale), P_nb, Gx_nb, Gy_nb)

        max_P  = max(max_P,  np.max(np.abs(P_nb  - P_sc )))
        max_Gx = max(max_Gx, np.max(np.abs(Gx_nb - Gx_sc)))
        max_Gy = max(max_Gy, np.max(np.abs(Gy_nb - Gy_sc)))

    print(f"  max |ΔP|       = {max_P:.2e}   (PSF peak = {psf.max():.4f})")
    print(f"  max |Δ(∂P/∂x)| = {max_Gx:.2e}")
    print(f"  max |Δ(∂P/∂y)| = {max_Gy:.2e}")
    print(f"  → fractional   = {max_P/psf.max():.2e}   [machine ε = 2.2e-16]")


# ---------------------------------------------------------------------------
# 2. FD gradient accuracy vs fine-step reference
# ---------------------------------------------------------------------------

def test_fd_accuracy(psf, psf_scale, label, n_positions=500, h_fine=0.01, rng=None):
    """Compare production h=1 ss px FD against h=0.01 ss px reference."""
    if rng is None:
        rng = np.random.default_rng(1)

    coeffs = spline_filter(psf, order=3, output=np.float64)
    size = psf.shape[0]
    half_ss = size // 2

    dix_1d = np.arange(-HW, HW + 1)
    diy_1d = np.arange(-HW, HW + 1)
    diy2d, dix2d = np.meshgrid(diy_1d, dix_1d, indexing='ij')

    dxs = rng.uniform(-0.5, 0.5, n_positions)
    dys = rng.uniform(-0.5, 0.5, n_positions)

    all_err  = []
    all_grad = []

    for dx, dy in zip(dxs, dys):
        xs_f = (half_ss + (dix2d.ravel() - dx) * psf_scale).astype(np.float64)
        ys_f = (half_ss + (diy2d.ravel() - dy) * psf_scale).astype(np.float64)
        kw = dict(order=3, mode='nearest', prefilter=False)

        Pxp = map_coordinates(coeffs, [ys_f, xs_f + 1.0], **kw)
        Pxm = map_coordinates(coeffs, [ys_f, xs_f - 1.0], **kw)
        Gx_prod = (Pxm - Pxp) * 0.5 * psf_scale

        Pxp_f = map_coordinates(coeffs, [ys_f, xs_f + h_fine], **kw)
        Pxm_f = map_coordinates(coeffs, [ys_f, xs_f - h_fine], **kw)
        Gx_ref = (Pxm_f - Pxp_f) / (2 * h_fine) * psf_scale

        all_err.append(np.abs(Gx_prod - Gx_ref))
        all_grad.append(np.abs(Gx_ref))

    all_err  = np.concatenate(all_err)
    all_grad = np.concatenate(all_grad)

    peak_grad  = all_grad.max()
    thresh_5pc = 0.05 * peak_grad
    mask_sig   = all_grad > thresh_5pc
    rel_err    = all_err[mask_sig] / all_grad[mask_sig]

    w_fisher = all_grad**2
    w_fisher /= w_fisher.sum()
    fisher_rms_err = np.sqrt(np.sum(w_fisher * all_err**2))

    print(f"  h_det = 1/psf_scale = {1/psf_scale:.4f} det px,  reference h = {h_fine} ss px")
    print(f"  actual peak |∂P/∂x| = {peak_grad:.5f}")
    print(f"  Absolute error on ∂P/∂x (all pixels):")
    print(f"    max  = {all_err.max():.4e}  ({all_err.max()/peak_grad*100:.2f}% of peak gradient)")
    print(f"    mean = {all_err.mean():.4e}")
    print(f"  Fisher-weighted RMS error (∝ position bias contribution):")
    print(f"    √(Σ w_fish · |ε|²) = {fisher_rms_err:.4e}  ({fisher_rms_err/peak_grad*100:.3f}% of peak)"  )
    print(f"  Relative error (where |∂P/∂x| > 5% of peak):")
    print(f"    max  = {rel_err.max()*100:.3f}%")
    print(f"    mean = {rel_err.mean()*100:.3f}%")


# ---------------------------------------------------------------------------
# 3. PSF spatial cache error
# ---------------------------------------------------------------------------

def test_cache_error(cube, xs, ys, psf_scale, n_test=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(3)

    x_lo, x_hi = xs[1], xs[-2]
    y_lo, y_hi = ys[1], ys[-2]

    x_test = rng.uniform(x_lo, x_hi, n_test)
    y_test = rng.uniform(y_lo, y_hi, n_test)
    
    # Simulate a cache cell size of 10 pixels (typical for jwst1pass)
    x_cell = (x_test // 10) * 10 + 5.0
    y_cell = (y_test // 10) * 10 + 5.0

    abs_errs  = []
    rel_errs  = []

    for xt, yt, xc, yc in zip(x_test, y_test, x_cell, y_cell):
        p_exact = interpolate_psf(cube, xs, ys, xt, yt)
        p_cache = interpolate_psf(cube, xs, ys, xc, yc)
        diff = np.abs(p_exact - p_cache)
        abs_errs.append(diff.max())
        rel_errs.append(diff.max() / p_exact.max())

    abs_errs = np.array(abs_errs)
    rel_errs = np.array(rel_errs)

    dx_cell = xs[1] - xs[0]
    print(f"  Grid: {len(xs)}×{len(ys)} PSFs,  spacing ~{dx_cell:.0f} px")
    print(f"  10-px cell = {10/dx_cell*100:.2f}% of grid spacing")
    print(f"  max  |PSF_exact − PSF_cached|          = {abs_errs.max():.4e}  (absolute)")
    print(f"  max  |PSF_exact − PSF_cached| / peak   = {rel_errs.max()*100:.4f}%")
    print(f"  mean |PSF_exact − PSF_cached| / peak   = {rel_errs.mean()*100:.4f}%")


# ---------------------------------------------------------------------------
# 4. Systematic position bias — noiseless injection
# ---------------------------------------------------------------------------

def test_systematic_bias(cube, xs, ys, psf_scale, label, n_phases=40, flux=50000.0):
    """Noise-free: inject at x_true, fit, measure |x_fit − x_true|."""
    x_det, y_det = xs[len(xs) // 2], ys[len(ys) // 2]
    x_true, y_true = 128.0, 128.0

    phases = np.linspace(-0.49, 0.49, n_phases)
    biases_x = []

    for phase in phases:
        x0 = x_true + phase
        inj_psf = interpolate_psf(cube, xs, ys, x_det, y_det)
        inj_coeffs = spline_filter(inj_psf, order=3, output=np.float64)
        image   = inject_noiseless(inj_coeffs, x0, y_true, flux, psf_scale)

        rec = fit_star(
            data=image, x0=x0, y0=y_true,
            psf_coeffs=inj_coeffs, psf_scale=psf_scale,
            hw=HW, sky_init=SKY, gain=GAIN, read_noise=RN,
            max_iter=50, tol=1e-7, zero_point=0.0, pass_number=1, sigma_clip=False,
        )
        if rec.converged:
            biases_x.append((phase, rec.x - x0))

    if not biases_x:
        print("  No converged fits — check parameters")
        return

    biases_x = np.array(biases_x)
    worst_i  = np.argmax(np.abs(biases_x[:, 1]))
    snr      = flux / np.sqrt(flux / GAIN + (RN / GAIN)**2)

    print(f"  flux={flux:.0f}, theoretical SNR≈{snr:.0f}, n_phases={n_phases}, tol=1e-7")
    print(f"  mean |x_fit − x_true| = {np.mean(np.abs(biases_x[:,1])):.6f} px")
    print(f"  max  |x_fit − x_true| = {np.max(np.abs(biases_x[:,1])):.6f} px   "
          f"(at phase={biases_x[worst_i,0]:.3f})")
    print(f"  rms  bias             = {np.sqrt(np.mean(biases_x[:,1]**2)):.6f} px")
    ok = np.max(np.abs(biases_x[:,1])) < 0.002 # Relaxed for JWST
    print(f"  → max systematic bias < 0.002 px? {'YES ✓' if ok else 'NO  ← ATTENTION'}")


# ---------------------------------------------------------------------------
# 5. Noisy MC — statistical precision floor
# ---------------------------------------------------------------------------

def test_noisy_mc(cube, xs, ys, psf_scale, n_phases=16, n_repeat=300, flux=10000.0):
    x_det, y_det = xs[len(xs) // 2], ys[len(ys) // 2]
    x_true, y_true = 128.0, 128.0

    inj_psf = interpolate_psf(cube, xs, ys, x_det, y_det)
    inj_coeffs = spline_filter(inj_psf, order=3, output=np.float64)

    rng = np.random.default_rng(77)
    phases = np.linspace(-0.45, 0.45, n_phases)
    mc_bias   = []
    mc_sigma  = []

    for phase in phases:
        x0 = x_true + phase
        image_clean = inject_noiseless(inj_coeffs, x0, y_true, flux, psf_scale)
        xs_fit = []

        for _ in range(n_repeat):
            noisy = rng.poisson(np.maximum(image_clean, 0.0)).astype(np.float64)
            noisy += rng.normal(0.0, RN, image_clean.shape)
            rec = fit_star(
                data=noisy, x0=x0, y0=y_true,
                psf_coeffs=inj_coeffs, psf_scale=psf_scale,
                hw=HW, sky_init=SKY, gain=GAIN, read_noise=RN,
                max_iter=25, tol=1e-5, zero_point=0.0, pass_number=1,
            )
            if rec.converged:
                xs_fit.append(rec.x - x0)

        if xs_fit:
            arr = np.array(xs_fit)
            mc_bias.append(np.mean(arr))
            mc_sigma.append(np.std(arr))

    mc_bias  = np.array(mc_bias)
    mc_sigma = np.array(mc_sigma)
    se       = mc_sigma.mean() / np.sqrt(n_repeat)
    snr      = flux / np.sqrt(flux / GAIN + (RN / GAIN)**2)

    print(f"  flux={flux:.0f}, SNR≈{snr:.0f}, n_phases={n_phases}×{n_repeat} draws")
    print(f"  position scatter (1-σ)   = {mc_sigma.mean():.5f} px")
    print(f"  max |MC phase mean bias| = {np.max(np.abs(mc_bias)):.5f} px")
    print(f"  std error of phase mean  = {se:.5f} px")
    sig = np.max(np.abs(mc_bias)) / se
    print(f"  → max bias = {sig:.1f}σ  "
          f"({'likely systematic' if sig > 3 else 'consistent with noise'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--psf',      type=str,   default=STDPSF_PATH)
    ap.add_argument('--flux',     type=float, default=50000.0)
    ap.add_argument('--mc-flux',  type=float, default=10000.0)
    ap.add_argument('--n-phases', type=int,   default=40)
    ap.add_argument('--n-repeat', type=int,   default=300)
    ap.add_argument('--skip-mc',  action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.psf):
        print(f"Error: PSF file not found: {args.psf}")
        sys.exit(1)

    print("=" * 65)
    print("jwst1pass PSF interpolation accuracy assessment")
    print(f"  PSF: {os.path.basename(args.psf)}")
    print(f"  Fit window: {2*HW+1}×{2*HW+1} px  (hw={HW})")
    print(f"  Numba available: {_NUMBA}")
    print("=" * 65)

    cube, xs, ys, psf_scale, grid_shape = load_psf_cube(args.psf)
    mid_idx = len(cube) // 2
    center_psf = cube[mid_idx].copy()
    gauss_psf = make_gaussian_psf(fwhm=2.0, scale=psf_scale)

    print(f"\nReal STDPSF: {cube.shape[0]} models, psf_scale={psf_scale}")
    print(f"  Grid: {len(xs)} x positions ({xs[0]:.0f}–{xs[-1]:.0f} px)")
    print(f"  Grid: {len(ys)} y positions ({ys[0]:.0f}–{ys[-1]:.0f} px)")
    print(f"  Centre PSF peak: {center_psf.max():.5f}")

    section("1. Numba vs scipy B-spline consistency")
    test_numba_scipy(center_psf, psf_scale, "real")
    test_numba_scipy(gauss_psf, psf_scale, "gauss")

    section("2. FD gradient accuracy  (h=1 ss px vs h=0.01 ss px reference)")
    test_fd_accuracy(center_psf, psf_scale, "real")
    test_fd_accuracy(gauss_psf, psf_scale, "gauss")

    section("3. PSF spatial grid interpolation error")
    test_cache_error(cube, xs, ys, psf_scale)

    section("4. Systematic position bias — noiseless injection")
    test_systematic_bias(cube, xs, ys, psf_scale, "real",
                         n_phases=args.n_phases, flux=args.flux)

    if not args.skip_mc:
        section("5. Noisy MC position precision")
        test_noisy_mc(cube, xs, ys, psf_scale,
                      n_phases=16, n_repeat=args.n_repeat, flux=args.mc_flux)
