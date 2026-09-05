"""Core PSF evaluation, sky estimation, source finding, and linear PSF fitter for JWST."""

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter, spline_filter
from dataclasses import dataclass, field
from astropy.stats import sigma_clipped_stats

# ---------------------------------------------------------------------------
# Numba bicubic B-spline kernel (optional fast path)
# ---------------------------------------------------------------------------
try:
    import numba as nb
    _NUMBA = True

    @nb.njit(cache=True, fastmath=True)
    def _bspline3(t):
        """Cubic B-spline basis function."""
        t = abs(t)
        if t < 1.0:
            return 2.0/3.0 + (0.5*t - 1.0)*t*t
        elif t < 2.0:
            s = 2.0 - t
            return s*s*s / 6.0
        return 0.0

    @nb.njit(cache=True, fastmath=True)
    def _bicubic_nearest(c, y, x):
        """Evaluate a 2-D cubic B-spline at (y, x), nearest boundary."""
        ny, nx = c.shape
        iy = int(np.floor(y)); ix = int(np.floor(x))
        v = 0.0
        for j in range(iy - 1, iy + 3):
            jj = 0 if j < 0 else (ny - 1 if j >= ny else j)
            wy = _bspline3(y - j)
            s = 0.0
            for i in range(ix - 1, ix + 3):
                ii = 0 if i < 0 else (nx - 1 if i >= nx else i)
                s += c[jj, ii] * _bspline3(x - i)
            v += wy * s
        return v

    @nb.njit(cache=True, fastmath=True)
    def _nb_eval_psf_grad(c, ys, xs, scale, P, dPdx, dPdy):
        """Evaluate PSF value + x/y gradients at all pixel positions in one pass."""
        n = len(ys)
        for k in range(n):
            y, x = ys[k], xs[k]
            P[k]    = _bicubic_nearest(c, y,       x)
            Pxp     = _bicubic_nearest(c, y,       x + 1.0)
            Pxm     = _bicubic_nearest(c, y,       x - 1.0)
            Pyp     = _bicubic_nearest(c, y + 1.0, x)
            Pym     = _bicubic_nearest(c, y - 1.0, x)
            dPdx[k] = (Pxm - Pxp) * 0.5 * scale
            dPdy[k] = (Pym - Pyp) * 0.5 * scale

except ImportError:
    _NUMBA = False


def _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale):
    """Evaluate PSF + gradients from pre-filtered B-spline coefficients."""
    size = coeffs.shape[0]
    half_ss = size // 2

    dix_bc, diy_bc = np.broadcast_arrays(np.asarray(dix), np.asarray(diy))
    shape = dix_bc.shape
    xs_flat = (half_ss + (dix_bc.ravel() - dx) * psf_scale).astype(np.float64)
    ys_flat = (half_ss + (diy_bc.ravel() - dy) * psf_scale).astype(np.float64)

    if _NUMBA:
        n = len(xs_flat)
        P_f    = np.empty(n, dtype=np.float64)
        dPdx_f = np.empty(n, dtype=np.float64)
        dPdy_f = np.empty(n, dtype=np.float64)
        _nb_eval_psf_grad(coeffs, ys_flat, xs_flat, float(psf_scale),
                          P_f, dPdx_f, dPdy_f)
        return P_f.reshape(shape), dPdx_f.reshape(shape), dPdy_f.reshape(shape)
    else:
        kw = dict(order=3, mode='nearest', prefilter=False)
        P    = map_coordinates(coeffs, [ys_flat, xs_flat    ], **kw).reshape(shape)
        Pxp  = map_coordinates(coeffs, [ys_flat, xs_flat + 1], **kw).reshape(shape)
        Pxm  = map_coordinates(coeffs, [ys_flat, xs_flat - 1], **kw).reshape(shape)
        Pyp  = map_coordinates(coeffs, [ys_flat + 1, xs_flat], **kw).reshape(shape)
        Pym  = map_coordinates(coeffs, [ys_flat - 1, xs_flat], **kw).reshape(shape)
        return P, (Pxm - Pxp) * 0.5 * psf_scale, (Pym - Pyp) * 0.5 * psf_scale


@dataclass
class StarRecord:
    x: float
    y: float
    flux: float
    flux_err: float
    sky: float
    sky_err: float
    mag: float
    mag_err: float
    qfit: float   # 5x5 window (small q)
    Qfit: float   # Full fit window (capital Q)
    chi2: float
    central_res: float
    n_sat: int = 0
    psf_frac: float = 0.0
    psf_peak: float = 0.0
    peak: float = 0.0
    cov: np.ndarray = None
    pass_number: int = 1
    n_neighbors: int = 0
    dist_nearest: float = np.inf
    dist_nearest_brighter: float = np.inf
    n_iter: int = 0
    converged: bool = False
    delta_max: float = 0.0
    chi2_scale: float = 1.0
    clipped_mask: np.ndarray = None
    concentration: float = field(default=np.nan)
    concentration_2x2: float = field(default=np.nan)
    concentration_3x3: float = field(default=np.nan)
    n_conc_1x1: int = 0
    n_conc_2x2: int = 0
    n_conc_3x3: int = 0
    is_star_candidate: bool = True


# ---------------------------------------------------------------------------
# Sky estimation
# ---------------------------------------------------------------------------

def estimate_sky_global(data, mask=None):
    """Estimate a single global sky value for the image using robust stats."""
    from astropy.stats import sigma_clipped_stats
    _, median, _ = sigma_clipped_stats(data, mask=mask, sigma=3.0)
    return float(median)


def estimate_sky(data, ix, iy, sky_inner, sky_outer, mask=None):
    """Sigma-clipped median sky from an annulus around integer pixel (ix, iy)."""
    ny, nx = data.shape
    y0 = max(0, iy - sky_outer)
    y1 = min(ny, iy + sky_outer + 1)
    x0 = max(0, ix - sky_outer)
    x1 = min(nx, ix + sky_outer + 1)

    sub = data[y0:y1, x0:x1]
    dy = np.arange(y0 - iy, y1 - iy)
    dx = np.arange(x0 - ix, x1 - ix)
    DY, DX = np.meshgrid(dy, dx, indexing='ij')
    r2 = DY**2 + DX**2

    in_ann = (r2 >= sky_inner**2) & (r2 <= sky_outer**2)
    if mask is not None:
        in_ann &= ~mask[y0:y1, x0:x1]

    pixels = sub[in_ann].astype(np.float64)
    if len(pixels) < 3:
        # Fallback to a robust median of the entire window if annulus is too small
        return float(np.nanmedian(sub)) if np.any(np.isfinite(sub)) else 0.0, 1.0

    # Explicitly filter for finite values to handle NaNs/Infs
    pixels = pixels[np.isfinite(pixels)]
    if len(pixels) < 3:
        return float(np.nanmedian(sub)) if np.any(np.isfinite(sub)) else 0.0, 1.0

    for _ in range(3):
        med = np.median(pixels)
        mad = np.median(np.abs(pixels - med))
        sigma = mad * 1.4826
        if sigma <= 0.0:
            break
        keep = np.abs(pixels - med) <= 3.0 * sigma
        if keep.sum() < 3:
            break
        pixels = pixels[keep]

    return float(np.median(pixels)), float(np.std(pixels))


def _build_annulus_offsets(sky_inner, sky_outer):
    """Return (iy_off, ix_off) 1D arrays for pixels in the sky annulus."""
    hw = sky_outer
    dy = np.arange(-hw, hw + 1)
    dx = np.arange(-hw, hw + 1)
    DY, DX = np.meshgrid(dy, dx, indexing='ij')
    r2 = DY**2 + DX**2
    ann = (r2 >= sky_inner**2) & (r2 <= sky_outer**2)
    iy_off, ix_off = np.where(ann)
    return (iy_off - hw).astype(np.int32), (ix_off - hw).astype(np.int32)


def estimate_sky_batch(data, ix_arr, iy_arr, sky_inner, sky_outer, mask=None):
    """Vectorised sigma-clipped median sky for many candidates simultaneously."""
    ny, nx = data.shape
    iy_off, ix_off = _build_annulus_offsets(sky_inner, sky_outer)

    y_pix = np.clip(iy_arr[:, None] + iy_off[None, :], 0, ny - 1)
    x_pix = np.clip(ix_arr[:, None] + ix_off[None, :], 0, nx - 1)

    pixels = data[y_pix, x_pix].astype(np.float64)

    if mask is not None:
        pixels[mask[y_pix, x_pix] != 0] = np.nan

    for _ in range(3):
        med = np.nanmedian(pixels, axis=1)
        dev = np.abs(pixels - med[:, None])
        mad = np.nanmedian(dev, axis=1)
        sigma = mad * 1.4826
        pixels[dev > 3.0 * sigma[:, None]] = np.nan

    sky_vals = np.nanmedian(pixels, axis=1)
    sky_sigs = np.nanstd(pixels, axis=1, ddof=0)

    global_sky = float(np.nanmedian(data))
    global_sig = float(np.nanstd(data) * 0.05)
    bad = ~np.isfinite(sky_vals)
    sky_vals[bad] = global_sky
    sky_sigs[bad] = global_sig

    return sky_vals, np.maximum(sky_sigs, 1e-10)


def eval_psf_and_grad(psf, dx, dy, dix, diy, psf_scale):
    """Evaluate PSF and x/y detector-pixel gradients over a pixel window."""
    from scipy.ndimage import spline_filter
    coeffs = spline_filter(psf, order=3, output=np.float64)
    return _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale)


def find_sources(data, sky_inner, sky_outer, hmin, fmin, mask=None,
                 suppress_radius=None):
    """Find point-source candidates as local maxima above the flux threshold.

    Returns
    -------
    xs, ys, peaks, skys, sigs : tuple of 1D float64 arrays, sorted by peak
        flux descending.  Empty arrays if no candidates found.
    """
    border = sky_outer + 1
    ny, nx = data.shape

    # Pre-mask data to prevent bad pixels from shadowing real stars.
    # Setting masked or invalid pixels to -inf ensures they cannot be local maxima
    # and cannot suppress dim nearby stars in the maximum_filter step.
    search_data = data.astype(np.float64).copy()
    if mask is not None:
        search_data[mask != 0] = -np.inf
    search_data[~np.isfinite(search_data)] = -np.inf

    # 8-neighbour local maxima (6x faster than maximum_filter(3))
    c = search_data[1:-1, 1:-1]
    is_max = np.zeros((ny, nx), dtype=bool)
    is_max[1:-1, 1:-1] = (
        (c > search_data[:-2, :-2]) & (c > search_data[:-2, 1:-1]) & (c > search_data[:-2, 2:]) &
        (c > search_data[1:-1, :-2]) & (c > search_data[1:-1, 2:]) &
        (c > search_data[2:, :-2])  & (c > search_data[2:, 1:-1]) & (c > search_data[2:, 2:])
    )

    is_max[:border, :]  = False;  is_max[-border:, :] = False
    is_max[:, :border]  = False;  is_max[:, -border:] = False

    if suppress_radius and suppress_radius > 1:
        dominant = maximum_filter(search_data, size=2 * suppress_radius + 1,
                                  mode='constant', cval=-np.inf)
        is_max &= (search_data == dominant)

    iy_arr, ix_arr = np.where(is_max)
    if len(iy_arr) == 0:
        return (np.empty(0), np.empty(0), np.empty(0),
                np.empty(0), np.empty(0))

    sky_vals, sky_sigs = estimate_sky_batch(
        data, ix_arr, iy_arr, sky_inner, sky_outer, mask)

    peaks = data[iy_arr, ix_arr] - sky_vals
    good  = (peaks >= fmin) & (sky_sigs > 0)

    ix_ok    = ix_arr[good].astype(np.float64)
    iy_ok    = iy_arr[good].astype(np.float64)
    peaks_ok = peaks[good]
    sky_ok   = sky_vals[good]
    sigs_ok  = sky_sigs[good]

    if len(ix_ok) == 0:
        return (np.empty(0), np.empty(0), np.empty(0),
                np.empty(0), np.empty(0))

    order = np.argsort(-peaks_ok)
    return (ix_ok[order], iy_ok[order], peaks_ok[order],
            sky_ok[order], sigs_ok[order])


def _window_offsets(xi, yi, hw, ny, nx):
    y_lo = max(0, yi - hw)
    y_hi = min(ny, yi + hw + 1)
    x_lo = max(0, xi - hw)
    x_hi = min(nx, xi + hw + 1)
    diy = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
    dix = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
    return y_lo, y_hi, x_lo, x_hi, diy, dix


def _failed_record(x0, y0, flux, sky, peak, pass_number, zero_point, flux_unit_conversion=1.0):
    from .utils import mag_from_flux
    # Scale flux back to physical units for magnitude reporting
    mag, mag_err = mag_from_flux(flux * flux_unit_conversion, np.inf, zero_point)
    return StarRecord(
        x=float(x0), y=float(y0), flux=float(flux), flux_err=np.inf,
        sky=float(sky), sky_err=np.inf,
        mag=float(mag), mag_err=np.inf,
        qfit=9.99, Qfit=9.99, chi2=np.inf, central_res=0.0,
        peak=float(peak), converged=False,
        n_iter=0, pass_number=pass_number
    )
def fit_star(data, x0, y0, psf_coeffs, psf_scale, hw, sky_init, gain, read_noise,
             max_iter=15, tol=1e-3, mask=None, noise_map=None,
             zero_point=0.0, pass_number=1, sigma_clip=True,
             sigma_clip_sigma=4.0, sigma_clip_iter=2, gain_map=None,
             read_noise_map=None, verbose=False, flux_unit_conversion=1.0):
    """Fit a single point source using a joint 4-parameter solver (flux, x, y, sky)."""
    ny, nx = data.shape
    xi0c = int(np.clip(round(x0), 0, nx - 1))
    yi0c = int(np.clip(round(y0), 0, ny - 1))
    peak0 = float(data[yi0c, xi0c]) - sky_init

    # Pre-solve for flux and sky with position fixed
    _dx0 = x0 - xi0c; _dy0 = y0 - yi0c
    _y0b, _y1b, _x0b, _x1b, _diy0, _dix0 = _window_offsets(xi0c, yi0c, hw, ny, nx)
    _d0 = data[_y0b:_y1b, _x0b:_x1b].astype(np.float64)
    _g0 = np.isfinite(_d0)
    if mask is not None:
        _g0 &= (mask[_y0b:_y1b, _x0b:_x1b] == 0)
    _g0 = _g0.ravel()
    _P0, _, _ = _eval_psf_grad_fast(psf_coeffs, _dx0, _dy0, _dix0, _diy0, psf_scale)
    _n0 = int(_g0.sum())
    if _n0 >= 2:
        _A2 = np.column_stack([_P0.ravel()[_g0], np.ones(_n0)])
        try:
            _fs, *_ = np.linalg.lstsq(_A2, _d0.ravel()[_g0], rcond=None)
            flux = max(float(_fs[0]), 1.0)
            sky = float(_fs[1])
        except np.linalg.LinAlgError:
            flux = max(peak0, 1.0)
            sky = sky_init
    else:
        flux = max(peak0, 1.0)
        sky = sky_init

    if verbose:
        print(f"DEBUG: Start fit at ({x0:.3f}, {y0:.3f}) flux={flux:.2e} sky={sky:.2f}")

    sky_initial = sky
    converged = False
    n_iter = max_iter
    clipped_mask = None

    for it in range(max_iter):
        if not (np.isfinite(x0) and np.isfinite(y0)):
            if verbose: print(f"DEBUG: Drifted to non-finite at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0 + sky_init, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)
            
        xi = int(round(x0))
        yi = int(round(y0))
        dx = x0 - xi
        dy = y0 - yi

        y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw, ny, nx)
        d = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        good = np.isfinite(d)
        if mask is not None:
            m_win = mask[y_lo:y_hi, x_lo:x_hi]
            good &= (m_win == 0)

        # Internal pixel quality flags (0=good, 1=clipped)
        # Ensure clipped_mask shape matches current window (can change if xi,yi shift near boundary)
        if clipped_mask is not None:
            if clipped_mask.shape == d.shape:
                good &= (clipped_mask == 0)
            else:
                # Reset if shape mismatch (rare, only if peak shifts across a boundary)
                clipped_mask = None

        if good.sum() < 5:
            if verbose: print(f"DEBUG: Not enough good pixels ({good.sum()}) at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)
            
        P, dPdx, dPdy = _eval_psf_grad_fast(psf_coeffs, dx, dy, dix, diy, psf_scale)
        
        if P.shape != d.shape:
            if verbose: print(f"DEBUG: Shape mismatch at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)
            
        r = d - sky - flux * P
        
        if noise_map is not None:
            # Pipeline ERR² already encodes stellar Poisson; add only the 1% systematic floor.
            # (Matches HST py1pass design: noise_map is used as-is, no self-Poisson added.)
            var = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
            var += (0.01 * flux * P)**2
        else:
            g_eff = gain_map[y_lo:y_hi, x_lo:x_hi] if gain_map is not None else gain
            # Use spatial read noise if available
            rn_eff = read_noise_map[y_lo:y_hi, x_lo:x_hi] if read_noise_map is not None else read_noise
            var = (np.maximum(flux * P, 0.0) + max(sky, 0.0)) / g_eff + (rn_eff / g_eff)**2
            var += (0.01 * flux * P)**2
        var = np.maximum(var, 1e-10)
        w = np.where(good, 1.0 / var, 0.0)
        
        g = good.ravel()
        w_g = w.ravel()[g]
        n_g = g.sum()
        
        A = np.column_stack([
            P.ravel()[g],
            (flux * dPdx).ravel()[g],
            (flux * dPdy).ravel()[g],
            np.ones(n_g)
        ])
        
        AtWA = A.T @ (w_g[:, None] * A)
        AtWr = A.T @ (w_g * r.ravel()[g])
        
        try:
            delta = np.linalg.solve(AtWA, AtWr)
        except np.linalg.LinAlgError:
            if verbose: print(f"DEBUG: LinAlgError at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)
            
        if not np.all(np.isfinite(delta)):
            if verbose: print(f"DEBUG: Non-finite delta at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)

        if verbose:
            print(f"DEBUG: it={it:2d} x={x0:8.3f} y={y0:8.3f} flux={flux:10.2e} sky={sky:8.2f} "
                  f"dx={delta[1]:8.4f} dy={delta[2]:8.4f} df={delta[0]:10.2e} ds={delta[3]:8.2f}")

        # Clamp position step: a shift larger than 2*hw pixels per iteration is
        # a sign of a diverging Newton step. Flag and bail out.
        # (Same as hst1pass_improved)
        if abs(delta[1]) > 2 * hw or abs(delta[2]) > 2 * hw:
            if verbose: print(f"DEBUG: Position clamp (>{2*hw}) at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)

        # Flux step: prevent flux from crashing more than 75 % in one iteration.
        # (Same as hst1pass_improved)
        _flux_floor = max(0.25 * flux, 1.0)
        if flux + delta[0] < _flux_floor:
            if verbose: print(f"DEBUG: Flux floor clamp at iter {it}")
            delta[0] = _flux_floor - flux

        flux += delta[0]
        x0 += delta[1]
        y0 += delta[2]
        sky += delta[3]
        flux = max(flux, 1.0)
        
        if x0 < 0 or x0 >= nx or y0 < 0 or y0 >= ny:
            if verbose: print(f"DEBUG: Out of bounds at iter {it}")
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)
        
        delta_max = max(abs(delta[1]), abs(delta[2]))
        if delta_max < tol:
            converged = True
            n_iter = it + 1
            break

    # --- Post-convergence sigma clipping ---
    # Iteratively mask pixels with large normalised residuals and re-solve.
    # Handles cosmic rays and warm pixels that survived the DQ mask.
    if converged and sigma_clip and sigma_clip_iter > 0:
        for _sc in range(sigma_clip_iter):
            # Check for non-finite values before each iteration
            if not (np.isfinite(x0) and np.isfinite(y0)):
                break

            # Re-evaluate residuals and variance at current solution.
            # Use _sc-suffixed locals so the post-Newton d/good/P/r/w remain intact.
            y_lo_sc, y_hi_sc, x_lo_sc, x_hi_sc, diy_sc, dix_sc = _window_offsets(
                int(round(x0)), int(round(y0)), hw, ny, nx)
            d_sc = data[y_lo_sc:y_hi_sc, x_lo_sc:x_hi_sc]
            P_sc, dPdx_sc, dPdy_sc = _eval_psf_grad_fast(
                psf_coeffs, x0-round(x0), y0-round(y0), dix_sc, diy_sc, psf_scale)
            r_sc = d_sc - sky - flux * P_sc

            # Recompute good mask for this window (shape may differ near edges)
            good_sc = np.isfinite(d_sc)
            if clipped_mask is not None and clipped_mask.shape == d_sc.shape:
                good_sc &= (clipped_mask == 0)

            # Hybrid model variance (matches main loop)
            g_eff = gain_map[y_lo_sc:y_hi_sc, x_lo_sc:x_hi_sc] if gain_map is not None else gain
            rn_eff = read_noise_map[y_lo_sc:y_hi_sc, x_lo_sc:x_hi_sc] if read_noise_map is not None else read_noise

            if noise_map is not None:
                var_sc = noise_map[y_lo_sc:y_hi_sc, x_lo_sc:x_hi_sc].astype(np.float64)
                var_sc += (0.01 * flux * P_sc)**2
            else:
                var_sc = (np.maximum(flux * P_sc, 0.0) + max(sky, 0.0)) / g_eff + (rn_eff / g_eff)**2
                var_sc += (0.01 * flux * P_sc)**2
            var_sc = np.maximum(var_sc, 1e-10)

            pull = np.abs(r_sc) / np.sqrt(var_sc)
            outlier = (pull > sigma_clip_sigma)
            new_good = good_sc & ~outlier

            if new_good.sum() < 5 or not outlier[good_sc].any():
                break

            g_sc_mask = new_good.ravel()
            w_sc = 1.0 / var_sc.ravel()[g_sc_mask]
            A_sc = np.column_stack([
                P_sc.ravel()[g_sc_mask],
                (flux * dPdx_sc).ravel()[g_sc_mask],
                (flux * dPdy_sc).ravel()[g_sc_mask],
                np.ones(int(g_sc_mask.sum()))
            ])
            try:
                delta_sc = np.linalg.solve(A_sc.T @ (w_sc[:, None] * A_sc),
                                           A_sc.T @ (w_sc * r_sc.ravel()[g_sc_mask]))
                # Guard against non-finite solutions
                if not np.all(np.isfinite(delta_sc)):
                    break
                if abs(delta_sc[1]) > 0.5 or abs(delta_sc[2]) > 0.5:
                    break
                if flux + delta_sc[0] < 1.0: delta_sc[0] = 1.0 - flux
                flux += delta_sc[0]; x0 += delta_sc[1]; y0 += delta_sc[2]; sky += delta_sc[3]
            except np.linalg.LinAlgError:
                break

    if not converged:
        pass

    # Final stats
    if converged:
        # 1. Calculate small qfit (q) over a central 5x5 "core" window to match Fortran
        # We re-optimize the flux locally for this window (zu = ptot/ftot)
        # This prevents qfit from being penalized by noise in the larger fit window.
        cy, cx = r.shape[0] // 2, r.shape[1] // 2
        y0_c, y1_c = max(0, cy - 2), min(r.shape[0], cy + 3)
        x0_c, x1_c = max(0, cx - 2), min(r.shape[1], cx + 3)
        
        # d_core is (data - sky)
        d_core = d[y0_c:y1_c, x0_c:x1_c] - sky
        good_core = good[y0_c:y1_c, x0_c:x1_c]
        P_core = P[y0_c:y1_c, x0_c:x1_c]
        
        if np.any(good_core):
            ptot = np.sum(d_core[good_core])
            ftot = np.sum(P_core[good_core])
            if ftot > 0:
                zu = ptot / ftot
                abstot = np.sum(np.abs(d_core[good_core] - zu * P_core[good_core]))
                qfit = abstot / max(ptot, 1e-10)
            else:
                qfit = 9.99
        else:
            qfit = 9.99

        # 2. Calculate capital Qfit (Q) over the full fit window
        d_full = d - sky
        good_full = good
        P_full = P
        
        if np.any(good_full):
            ptot_full = np.sum(d_full[good_full])
            ftot_full = np.sum(P_full[good_full])
            if ftot_full > 0:
                # Local flux re-optimization for full window
                zu_full = ptot_full / ftot_full
                abstot_full = np.sum(np.abs(d_full[good_full] - zu_full * P_full[good_full]))
                Qfit = abstot_full / max(ptot_full, 1e-10)
            else:
                Qfit = 9.99
        else:
            Qfit = 9.99

        if qfit > 9.99: qfit = 9.99
        if Qfit > 9.99: Qfit = 9.99

        # Store root reduced chi2 for consistency with HST
        # We use 4 DOF (flux, x, y, sky) as we fit the sky simultaneously.
        red_chi2 = np.sum(w[good] * r[good]**2) / max(1, good.sum() - 4)
        chi2 = float(np.sqrt(max(red_chi2, 0.0)))
        
        try:
            cov = np.linalg.inv(AtWA)
        except np.linalg.LinAlgError:
            return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)
            
        # Peak PSF value in window (for normalization)
        psf_peak = np.max(P)
        
        # Saturated pixels in fit window
        n_sat = 0
        if mask is not None:
            n_sat = np.sum(mask[y_lo:y_hi, x_lo:x_hi] != 0)

        # ── Concentration metrics ─────────────────────────────────────────────
        # Ratio of observed pixel sum to PSF-model-predicted sum over three
        # aperture sizes.  Stars ≈ 1.0; extended sources < 1; CRs > 1.
        # Only pixels that are good (DQ-valid, shape-ok) enter both numerator
        # and denominator so masked pixels don't bias the ratio.
        xi_f = int(np.clip(round(x0), 0, nx - 1))
        yi_f = int(np.clip(round(y0), 0, ny - 1))

        def _pixel_good_local(abs_y, abs_x):
            ly = abs_y - y_lo; lx = abs_x - x_lo
            return (0 <= ly < good.shape[0]) and (0 <= lx < good.shape[1]) and bool(good[ly, lx])

        _peak_val = float(data[yi_f, xi_f]) - sky
        _conc_denom = float(flux) * max(float(psf_peak), 1e-10)
        _cen_good = _pixel_good_local(yi_f, xi_f)
        concentration = (float(_peak_val / _conc_denom)
                         if _conc_denom > 0 and _cen_good else np.nan)
        n_conc_1x1 = int(_cen_good)

        # 2×2 bracketing the fitted sub-pixel position
        _ix_lo = xi_f + (0 if (x0 - xi_f) >= 0 else -1)
        _iy_lo = yi_f + (0 if (y0 - yi_f) >= 0 else -1)
        _dix_2 = np.array([[_ix_lo - xi_f, _ix_lo + 1 - xi_f],
                            [_ix_lo - xi_f, _ix_lo + 1 - xi_f]])
        _diy_2 = np.array([[_iy_lo - yi_f, _iy_lo - yi_f],
                            [_iy_lo + 1 - yi_f, _iy_lo + 1 - yi_f]])
        _y2_abs = np.array([_iy_lo, _iy_lo, _iy_lo + 1, _iy_lo + 1])
        _x2_abs = np.array([_ix_lo, _ix_lo + 1, _ix_lo, _ix_lo + 1])
        _good_2 = np.array([_pixel_good_local(int(_y2_abs[k]), int(_x2_abs[k])) for k in range(4)])
        _y2 = np.clip(_y2_abs, 0, ny - 1).astype(int)
        _x2 = np.clip(_x2_abs, 0, nx - 1).astype(int)
        _d2 = data[_y2, _x2].astype(np.float64) - sky
        _P2, _, _ = _eval_psf_grad_fast(psf_coeffs, x0 - xi_f, y0 - yi_f,
                                         _dix_2, _diy_2, psf_scale)
        _P2_sum = float(_P2.ravel()[_good_2].sum())
        n_conc_2x2 = int(_good_2.sum())
        concentration_2x2 = (float(_d2[_good_2].sum() / (flux * _P2_sum))
                             if n_conc_2x2 >= 2 and _P2_sum > 0 and flux > 0 else np.nan)

        # 3×3 centred at the rounded position
        _off = np.array([-1, 0, 1])
        _DIX_3, _DIY_3 = np.meshgrid(_off, _off)
        _y3_abs = yi_f + _DIY_3.ravel()
        _x3_abs = xi_f + _DIX_3.ravel()
        _good_3 = np.array([_pixel_good_local(int(_y3_abs[k]), int(_x3_abs[k])) for k in range(9)])
        _y3 = np.clip(_y3_abs, 0, ny - 1).astype(int)
        _x3 = np.clip(_x3_abs, 0, nx - 1).astype(int)
        _d3 = data[_y3, _x3].astype(np.float64) - sky
        _P3, _, _ = _eval_psf_grad_fast(psf_coeffs, x0 - xi_f, y0 - yi_f,
                                         _DIX_3, _DIY_3, psf_scale)
        _P3_sum = float(_P3.ravel()[_good_3].sum())
        n_conc_3x3 = int(_good_3.sum())
        concentration_3x3 = (float(_d3[_good_3].sum() / (flux * _P3_sum))
                             if n_conc_3x3 >= 5 and _P3_sum > 0 and flux > 0 else np.nan)

    else:
        return _failed_record(x0, y0, flux, sky, peak0, pass_number, zero_point, flux_unit_conversion=flux_unit_conversion)

    from .utils import mag_from_flux
    mag, mag_err = mag_from_flux(flux * flux_unit_conversion, 0.0, zero_point)  # err filled later by inflation

    return StarRecord(
        x=float(x0), y=float(y0), flux=float(flux), flux_err=0.0,
        sky=float(sky), sky_err=0.0,
        mag=float(mag), mag_err=0.0,
        qfit=float(qfit), Qfit=float(Qfit), chi2=float(chi2), central_res=0.0,
        peak=float(peak0 + sky_init), psf_peak=float(psf_peak),
        n_sat=int(n_sat), cov=cov,
        converged=converged, n_iter=n_iter, pass_number=pass_number,
        clipped_mask=clipped_mask,
        concentration=concentration,
        concentration_2x2=concentration_2x2,
        concentration_3x3=concentration_3x3,
        n_conc_1x1=n_conc_1x1,
        n_conc_2x2=n_conc_2x2,
        n_conc_3x3=n_conc_3x3,
    )


def _build_chi2_mag_bins(mags, chi2s, bin_width=0.5, min_bin_width=0.1,
                         min_stars=50, max_stars=200, smooth_bandwidth=0.75):
    """Bin chi² by adaptive magnitude bins, merge small bins, then smooth.

    Used by both inflate_chi2 (covariance scaling) and plot_catalog_stats
    (overlay curve) so both always display the same correction.

    Returns 8-tuple:
        (bin_mags, bin_smooth, bin_raw, bin_unc,
         extrap_mags, extrap_chi2, extrap_faint_mags, extrap_faint_chi2)
    or None if fewer than 2 bins result.
    """
    try:
        from scipy.stats import median_abs_deviation as _madfn
    except ImportError:
        def _madfn(x, scale='normal'):
            med = np.median(x)
            m = np.median(np.abs(x - med))
            return m * (1.4826 if scale == 'normal' else 1.0)

    order   = np.argsort(mags)
    mags_s  = mags[order]
    chi2s_s = chi2s[order]

    init_bins = []
    bin_start = 0
    for i in range(len(mags_s)):
        n_in_bin = i - bin_start
        span     = mags_s[i] - mags_s[bin_start] if n_in_bin > 0 else 0.0
        hit_max  = n_in_bin >= max_stars or span >= bin_width
        if n_in_bin > 0 and hit_max and span >= min_bin_width:
            init_bins.append((mags_s[bin_start:i], chi2s_s[bin_start:i]))
            bin_start = i
    if bin_start < len(mags_s):
        init_bins.append((mags_s[bin_start:], chi2s_s[bin_start:]))

    if not init_bins:
        return None

    carry_m = carry_c = np.array([])
    merged  = []
    for m, c in init_bins:
        cm = np.concatenate([carry_m, m])
        cc = np.concatenate([carry_c, c])
        if len(cm) >= min_stars:
            merged.append((cm, cc))
            carry_m = carry_c = np.array([])
        else:
            carry_m, carry_c = cm, cc
    if carry_m.size:
        if merged:
            mm, mc = merged[-1]
            merged[-1] = (np.concatenate([mm, carry_m]),
                          np.concatenate([mc, carry_c]))
        else:
            merged.append((carry_m, carry_c))

    if len(merged) < 2:
        return None

    bin_mags = np.array([float(np.median(m)) for m, _ in merged])
    bin_raw  = np.clip([float(np.median(c)) for _, c in merged], 0.1, 20.0)
    bin_raw  = np.asarray(bin_raw, dtype=float)
    bin_unc  = []
    for _, c in merged:
        n = len(c)
        if n >= 3:
            mad = float(_madfn(c, scale='normal'))
            sigma_med = 1.2533 * mad / np.sqrt(n)
        elif n == 2:
            sigma_med = float(np.abs(c[1] - c[0])) / np.sqrt(2)
        else:
            sigma_med = 0.5
        bin_unc.append(max(sigma_med, 0.01))
    bin_unc = np.array(bin_unc)
    n_bins  = len(bin_raw)

    n_fit_l = min(3, n_bins)
    if n_fit_l >= 2:
        _c_bright = np.polyfit(np.arange(n_fit_l, dtype=float), bin_raw[:n_fit_l], 1,
                               w=1.0 / bin_unc[:n_fit_l])
        if _c_bright[0] > 0:
            _c_bright = np.array([0.0, float(bin_raw[0])])
    else:
        _c_bright = None

    n_fit_r = min(3, n_bins)
    faint_x = np.arange(n_bins - n_fit_r, n_bins, dtype=float)
    if n_fit_r >= 2:
        _c_faint = np.polyfit(faint_x, bin_raw[-n_fit_r:], 1,
                              w=1.0 / bin_unc[-n_fit_r:])
        if _c_faint[0] < 0:
            _c_faint = np.array([0.0, float(bin_raw[-1])])
    else:
        _c_faint = None

    if _c_bright is not None and n_fit_l >= 2:
        dM = float(np.median(np.diff(bin_mags[:n_fit_l])))
        if dM > 0:
            gap = bin_mags[0] - float(mags_s[0])
            n_extrap = max(2, int(np.ceil(gap / dM)) + 1)
            extrap_idx  = np.arange(-n_extrap, 0, dtype=float)
            extrap_mags = bin_mags[0] + extrap_idx * dM
            extrap_chi2 = np.maximum(np.polyval(_c_bright, extrap_idx), 1.0)
        else:
            extrap_mags = extrap_chi2 = np.array([])
    else:
        extrap_mags = extrap_chi2 = np.array([])

    extrap_faint_mags = extrap_faint_chi2 = np.array([])

    weights = 1.0 / (bin_unc ** 2)
    n_pad   = max(5, int(np.ceil(3.0 * smooth_bandwidth)))

    if _c_bright is not None:
        pad_left = np.maximum(np.polyval(_c_bright, np.arange(-n_pad, 0, dtype=float)), 1.0)
    else:
        pad_left = np.full(n_pad, max(float(bin_raw[0]), 1.0))
    if _c_faint is not None:
        pad_right = np.maximum(
            np.polyval(_c_faint, np.arange(n_bins, n_bins + n_pad, dtype=float)), 1.0)
    else:
        pad_right = np.full(n_pad, float(bin_raw[-1]))

    idx_full = np.concatenate([
        np.arange(-n_pad, 0),
        np.arange(n_bins),
        np.arange(n_bins, n_bins + n_pad),
    ], dtype=float)
    pv = np.concatenate([pad_left,                        bin_raw,  pad_right])
    pw = np.concatenate([np.full(n_pad, weights[0]),      weights,  np.full(n_pad, weights[-1])])

    smoothed = np.empty_like(bin_raw)
    for i in range(n_bins):
        d           = (idx_full - i) / smooth_bandwidth
        kw          = pw * np.exp(-0.5 * d * d)
        smoothed[i] = np.sum(kw * pv) / np.sum(kw)

    smoothed = np.clip(smoothed, 0.1, 20.0)
    return (bin_mags, smoothed, bin_raw, bin_unc,
            extrap_mags, extrap_chi2,
            extrap_faint_mags, extrap_faint_chi2)


def inflate_chi2(records, zero_point=0.0, exposure_time=1.0, verbose=False):
    """Apply magnitude-dependent chi²-scaling to covariances across *records* in-place.

    Builds the correction curve via _build_chi2_mag_bins (adaptive magnitude
    bins + Gaussian smoothing + Pchip interpolation) — the same function used
    by plot_catalog_stats to draw the overlay.  This ensures the plotted curve
    exactly matches what was applied to the covariances.

    The scaling is applied regardless of sign (chi2_eff > 1 inflates, < 1
    deflates), so overestimated noise is also corrected downward.

    Returns a dict with the correction curve for diagnostic use:
        {'bin_mags': 1D array, 'bin_chi2': 1D array,
         'extrap_mags': 1D array, 'extrap_chi2': 1D array, 'n_bins': int}
    """
    from .utils import mag_from_flux as _mff

    useful_mags  = []
    useful_chi2s = []
    for r in records:
        if (np.isfinite(r.chi2) and np.isfinite(r.mag)
                and r.qfit < 2.0 and r.chi2 < 10.0 and r.flux > 1.1
                and getattr(r, 'is_star_candidate', True)):
            useful_mags.append(r.mag)
            useful_chi2s.append(r.chi2)

    correction_info = {'bin_mags': np.array([]), 'bin_chi2': np.array([]),
                       'extrap_mags': np.array([]), 'extrap_chi2': np.array([]),
                       'n_bins': 0}

    _pchip     = None
    _pchip_mags = None
    _pchip_chi2 = None

    if len(useful_mags) >= 10:
        mags_u  = np.array(useful_mags)
        chi2s_u = np.array(useful_chi2s)
        result  = _build_chi2_mag_bins(mags_u, chi2s_u)

        if result is not None:
            (bin_mags, bin_smooth, bin_raw, bin_unc,
             extrap_mags, extrap_chi2,
             extrap_faint_mags, extrap_faint_chi2) = result

            # Build extended magnitude axis for Pchip (includes extrapolation anchors)
            pm = bin_mags.copy()
            pc = bin_smooth.copy()
            if extrap_mags.size:
                pm = np.concatenate([extrap_mags,       pm])
                pc = np.concatenate([extrap_chi2,       pc])
            if extrap_faint_mags.size:
                pm = np.concatenate([pm, extrap_faint_mags])
                pc = np.concatenate([pc, extrap_faint_chi2])

            try:
                from scipy.interpolate import PchipInterpolator as _Pchip
                _pchip      = _Pchip(pm, pc)
                _pchip_mags = pm
                _pchip_chi2 = pc
            except Exception:
                pass

            correction_info = {
                'bin_mags':    bin_mags,
                'bin_chi2':    bin_smooth,
                'extrap_mags': extrap_mags,
                'extrap_chi2': extrap_chi2,
                'n_bins':      len(bin_mags),
            }
            if verbose:
                print(f"  chi²_scale: {len(bin_mags)} mag bins, "
                      f"range [{bin_smooth.min():.3f}, {bin_smooth.max():.3f}]")

    if _pchip is None:
        # Fallback: global median
        chi2_global = float(np.median(useful_chi2s)) if useful_chi2s else 1.0
        if verbose:
            print(f"  chi²_global (too few stars or binning failed): {chi2_global:.3f}")
        _pchip_mags = np.array([-99.0, 99.0])
        _pchip_chi2 = np.array([chi2_global, chi2_global])

    def _eval_chi2_eff(mag_val):
        if _pchip is not None:
            if mag_val <= _pchip_mags[0]:
                return float(_pchip_chi2[0])
            if mag_val >= _pchip_mags[-1]:
                return float(_pchip_chi2[-1])
            return float(np.clip(_pchip(mag_val), 0.1, 20.0))
        # linear interpolation fallback (when pchip failed)
        return float(np.interp(mag_val, _pchip_mags, _pchip_chi2))

    for r in records:
        if not np.isfinite(r.chi2):
            r.chi2_scale = 1.0
        elif r.qfit >= 9.0:
            r.chi2_scale = float(r.chi2)
        else:
            chi2_eff = max(_eval_chi2_eff(r.mag), 0.1)
            chi2_residual = r.chi2 / chi2_eff
            # Use the star's own chi² when it is far above its peers — keeps
            # outlier stars from having artificially tight covariances.
            r.chi2_scale = float(r.chi2 if chi2_residual > 1.5 else chi2_eff)

        total_scale = r.chi2_scale
        if r.cov is not None:
            r.cov      *= (total_scale ** 2)
            r.flux_err  = float(np.sqrt(max(r.cov[0, 0], 0.0)))
            r.sky_err   = float(np.sqrt(max(r.cov[3, 3], 0.0)))
            _, r.mag_err = _mff(r.flux, r.flux_err, zero_point)

    return correction_info


# ---------------------------------------------------------------------------
# Star/galaxy classification
# ---------------------------------------------------------------------------

def _conc_adaptive_bounds(conc_arr, mag_arr, star_mask,
                          fixed_lo, fixed_hi,
                          mag_bin_width, min_bin_stars,
                          conc_width_factor, conc_min_width,
                          conc_hard_lo=0.5, conc_hard_hi=2.0):
    """Compute per-star adaptive concentration bounds from the stellar locus.

    Uses the binned median ± conc_width_factor × half-width of the 68% region
    (with a floor of conc_min_width on the half-width) among sources in
    *star_mask*.  Falls back to the fixed [fixed_lo, fixed_hi] bounds where
    there are too few stars in a bin or overall.

    Returns
    -------
    lo_per_star, hi_per_star : ndarray, same length as conc_arr
    """
    n = len(conc_arr)
    lo_per_star = np.full(n, fixed_lo)
    hi_per_star = np.full(n, fixed_hi)

    avail = np.isfinite(conc_arr) & np.isfinite(mag_arr)
    sm = star_mask & avail & (conc_arr >= conc_hard_lo) & (conc_arr <= conc_hard_hi)
    if sm.sum() < min_bin_stars:
        return lo_per_star, hi_per_star

    mag_s  = mag_arr[sm]
    conc_s = conc_arr[sm]
    bin_edges   = np.arange(mag_s.min(), mag_s.max() + mag_bin_width, mag_bin_width)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bmed = np.full(len(bin_centres), np.nan)
    blo  = np.full(len(bin_centres), np.nan)
    bhi  = np.full(len(bin_centres), np.nan)
    for k, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        m = (mag_s >= lo) & (mag_s < hi)
        if m.sum() >= min_bin_stars:
            v  = conc_s[m]
            med = float(np.median(v))
            hw  = max(float((np.percentile(v, 84) - np.percentile(v, 16)) / 2),
                      conc_min_width)
            bmed[k] = med
            blo[k]  = med - conc_width_factor * hw
            bhi[k]  = med + conc_width_factor * hw

    ok = np.isfinite(bmed)
    if ok.sum() < 2:
        return lo_per_star, hi_per_star

    bc_ok = bin_centres[ok]
    lo_per_star = np.interp(mag_arr, bc_ok, blo[ok])
    hi_per_star = np.interp(mag_arr, bc_ok, bhi[ok])
    return lo_per_star, hi_per_star


def classify_stars(records,
                   conc_lo=0.75, conc_hi=None,
                   qfit_global_max=1.5,
                   qfit_percentile=25, qfit_multiplier=4.0,
                   mag_bin_width=0.5, min_bin_stars=5,
                   conc_width_factor=4.0, conc_min_width=0.01,
                   conc_hard_lo=0.5, conc_hard_hi=2.0,
                   max_conc_iter=10):
    """Classify each record as a likely star or likely non-star.

    Sets ``is_star_candidate`` on each record in-place based on two criteria:

    1. **Concentration** (two-pass adaptive):
       Pass 1 seeds the stellar locus using the fixed [conc_lo, 1/conc_lo]
       window.  Pass 2 replaces the fixed window with per-magnitude bounds of
       ``median ± conc_width_factor × half_width`` (where half_width is half
       the 68% spread, floored at conc_min_width).  All three metrics that have
       finite values must pass; NaN metrics are skipped.

    2. **qfit**: below both a global cap (*qfit_global_max*) and a
       magnitude-adaptive threshold derived from the p<qfit_percentile> of
       qfit within each magnitude bin, multiplied by *qfit_multiplier*.

    Parameters
    ----------
    conc_lo          : float — initial lower bound; upper = 1/conc_lo
    conc_width_factor: float — adaptive half-width multiplier (default 4)
    conc_min_width   : float — floor on adaptive half-width (default 0.05)
    qfit_global_max  : float — hard upper limit regardless of adaptive threshold
    qfit_percentile  : int   — percentile of qfit used as the locus anchor
    qfit_multiplier  : float — multiplier above the percentile → threshold
    mag_bin_width    : float — magnitude bin width for adaptive thresholds
    min_bin_stars    : int   — minimum stars per bin to compute local threshold

    Returns
    -------
    n_candidates : int — number of records flagged as likely stars
    """
    if conc_hi is None:
        conc_hi = 1.0 / conc_lo

    if not records:
        return 0

    mag    = np.array([r.mag   for r in records])
    qfit   = np.array([r.qfit  for r in records])
    conc1  = np.array([getattr(r, 'concentration',      np.nan) for r in records])
    conc2  = np.array([getattr(r, 'concentration_2x2',  np.nan) for r in records])
    conc3  = np.array([getattr(r, 'concentration_3x3',  np.nan) for r in records])

    _a1 = np.isfinite(conc1)
    _a2 = np.isfinite(conc2)
    _a3 = np.isfinite(conc3)
    _kw = dict(mag_bin_width=mag_bin_width, min_bin_stars=min_bin_stars,
               conc_width_factor=conc_width_factor, conc_min_width=conc_min_width,
               conc_hard_lo=conc_hard_lo, conc_hard_hi=conc_hard_hi)

    def _conc_check(lo1, hi1, lo2, hi2, lo3, hi3):
        p1 = ~_a1 | ((conc1 >= lo1) & (conc1 <= hi1))
        p2 = ~_a2 | ((conc2 >= lo2) & (conc2 <= hi2))
        p3 = ~_a3 | ((conc3 >= lo3) & (conc3 <= hi3))
        return p1 & p2 & p3

    # Seed — 2×2 only with fixed [conc_lo, conc_hi].
    current_star = ~_a2 | ((conc2 >= conc_lo) & (conc2 <= conc_hi))

    # Global qfit cap.
    qfit_ok = np.isfinite(qfit) & (qfit < qfit_global_max)

    # Adaptive magnitude-bin qfit threshold: trace the stellar qfit locus.
    finite_mask = np.isfinite(mag) & np.isfinite(qfit)
    if finite_mask.sum() >= min_bin_stars:
        mag_f   = mag[finite_mask]
        qfit_f  = qfit[finite_mask]
        mag_min = float(np.nanmin(mag_f))
        mag_max = float(np.nanmax(mag_f))
        bin_edges = np.arange(mag_min, mag_max + mag_bin_width, mag_bin_width)

        bin_centers    = []
        bin_thresholds = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (mag_f >= lo) & (mag_f < hi)
            if m.sum() >= min_bin_stars:
                p_qfit = float(np.percentile(qfit_f[m], qfit_percentile))
                thresh = min(p_qfit * qfit_multiplier, qfit_global_max)
                bin_centers.append(float((lo + hi) / 2.0))
                bin_thresholds.append(thresh)

        if len(bin_centers) >= 2:
            bc = np.array(bin_centers)
            bt = np.array(bin_thresholds)
            adaptive_thresh = np.interp(mag, bc, bt, left=bt[0], right=bt[-1])
            qfit_ok &= (qfit < adaptive_thresh)

    current_star = current_star & qfit_ok

    # Iterate: fit adaptive bounds from current_star (all 3 metrics), re-select,
    # repeat until convergence.  At convergence the final star set is exactly
    # the set of sources within the bounds derived from itself — self-consistent
    # by construction, so the plot can use is_star_candidate directly.
    for _ in range(max_conc_iter):
        _lo1, _hi1 = _conc_adaptive_bounds(conc1, mag, current_star, conc_lo, conc_hi, **_kw)
        _lo2, _hi2 = _conc_adaptive_bounds(conc2, mag, current_star, conc_lo, conc_hi, **_kw)
        _lo3, _hi3 = _conc_adaptive_bounds(conc3, mag, current_star, conc_lo, conc_hi, **_kw)
        new_star = _conc_check(_lo1, _hi1, _lo2, _hi2, _lo3, _hi3) & qfit_ok
        if np.array_equal(new_star, current_star):
            break
        current_star = new_star

    n_candidates = 0
    for i, r in enumerate(records):
        r.is_star_candidate = bool(current_star[i])
        if r.is_star_candidate:
            n_candidates += 1

    return n_candidates


def compute_neighbor_stats(records, hw):
    """Compute per-star neighbour metrics and store them on each StarRecord."""
    n = len(records)
    if n == 0:
        return

    from scipy.spatial import cKDTree

    xy = np.array([[r.x, r.y] for r in records], dtype=np.float64)
    flux = np.array([r.flux for r in records], dtype=np.float64)

    if n == 1:
        records[0].n_neighbors = 0
        records[0].dist_nearest = np.inf
        records[0].dist_nearest_brighter = np.inf
        return

    tree = cKDTree(xy)

    try:
        counts = tree.query_ball_point(xy, r=hw, return_length=True)
    except TypeError:
        counts = np.array([len(lst) for lst in tree.query_ball_point(xy, r=hw)])

    K = min(n, 65)
    dists, idxs = tree.query(xy, k=K)

    for i, rec in enumerate(records):
        rec.n_neighbors = int(counts[i]) - 1
        rec.dist_nearest = float(dists[i, 1]) if n >= 2 else np.inf

        neigh_flux = flux[idxs[i, 1:]]
        neigh_dist = dists[i, 1:]
        brighter = neigh_flux > flux[i]
        rec.dist_nearest_brighter = (
            float(neigh_dist[brighter].min()) if brighter.any() else np.inf
        )
