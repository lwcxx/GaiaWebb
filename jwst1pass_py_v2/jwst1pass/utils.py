"""Astronomical utilities for JWST photometry."""

import numpy as np

def mag_from_flux(flux, flux_err, zero_point=0.0):
    """
    Return (mag, mag_err) from flux and uncertainty.
    Follows legacy Fortran convention: mag = -2.5 * log10(Flux_Image).
    """
    f_eff = float(flux)
    if not (f_eff > 0.01):
        f_eff = 0.01
    if not (f_eff < 1e19) or np.isnan(f_eff):
        # In case of extreme overflow or NaN, use a very small flux 
        # to ensure it's flagged as a poor fit (instrumental mag 99.99)
        f_eff = 10**((zero_point - 99.99)/2.5)
        
    # Instrumental magnitude: zp - 2.5 * log10(flux)
    mag = zero_point - 2.5 * np.log10(f_eff)
    
    if np.isfinite(flux_err) and flux_err > 0:
        # Propagation of error: dmag = 1.0857 * df/f
        mag_err = (1.0857) * flux_err / f_eff
    else:
        mag_err = np.inf
        
    return float(mag), float(mag_err)


def mag_to_flux(mag, zero_point=0.0, exposure_time=1.0):
    """
    Convert magnitude back to total flux.
    If exposure_time > 1, assumes mag is based on count rate.
    """
    # flux / exptime = 10**((zp - mag)/2.5)
    return exposure_time * 10**((zero_point - mag) / 2.5)


def _catmull_rom_weights(t):
    """Calculate Catmull-Rom interpolation weights for a 4-point stencil."""
    t2 = t * t
    t3 = t2 * t
    return np.array([
        -0.5*t3 + t2 - 0.5*t,
         1.5*t3 - 2.5*t2 + 1.0,
        -1.5*t3 + 2.0*t2 + 0.5*t,
         0.5*t3 - 0.5*t2
    ])


def interpolate_psf(psf_cube, xs, ys, x_det, y_det, _cache=None):
    """Interpolate the PSF grid to a specific detector position using Catmull-Rom.

    _cache : dict or None
        Optional per-run cache.  Keys are (x//5, y//5); PSF variation over a
        5 px cell is negligible for photometry.  Max 2048 entries.
    """
    ny_g, nx_g = len(ys), len(xs)

    if ny_g == 1 and nx_g == 1:
        return psf_cube[0]

    if _cache is not None:
        key = (int(x_det) // 5, int(y_det) // 5)
        hit = _cache.get(key)
        if hit is not None:
            return hit

    # Normalized grid coordinates
    gx = np.interp(x_det, xs, np.arange(nx_g))
    gy = np.interp(y_det, ys, np.arange(ny_g))

    ix = int(min(np.floor(gx), nx_g - 2))
    iy = int(min(np.floor(gy), ny_g - 2))
    tx = gx - ix
    ty = gy - iy

    wx = _catmull_rom_weights(tx)
    wy = _catmull_rom_weights(ty)

    # 4x4 stencil
    iy_idx = np.clip(np.arange(iy - 1, iy + 3), 0, ny_g - 1)
    ix_idx = np.clip(np.arange(ix - 1, ix + 3), 0, nx_g - 1)

    flat_idx = (iy_idx[:, None] * nx_g + ix_idx[None, :]).ravel()
    psf_block = psf_cube[flat_idx].reshape(4, 4, *psf_cube.shape[1:])

    weights = np.outer(wy, wx)
    result = np.einsum('ij,ij...->...', weights, psf_block)

    if _cache is not None and len(_cache) < 2048:
        _cache[key] = result
    return result
