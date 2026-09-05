"""Multi-pass photometry coordinator for JWST."""

import numpy as np
from joblib import Parallel, delayed
from .core import fit_star, inflate_chi2, estimate_sky, estimate_sky_global, find_sources, _eval_psf_grad_fast, _window_offsets, compute_neighbor_stats, classify_stars
from .utils import interpolate_psf, mag_to_flux

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sub_hw(psf_cube, psf_scale):
    """Half-width in detector pixels covering the full PSF array for subtraction."""
    psf_size = psf_cube.shape[-1]
    return (psf_size - 1) // (2 * int(psf_scale))


def _should_subtract(rec):
    """True if this star's PSF model should be removed from the residual image."""
    return (rec.qfit < 2.0 and
            rec.chi2 < 10.0 and
            getattr(rec, 'converged', True) and
            rec.flux > 0)


def _psf_window(rec, cube, xs, ys, psf_scale, hw, ny, nx,
                prefiltered, x_offset=0.0, y_offset=0.0, 
                psf_cache=None, hw_override=None):
    """Return (P, y_lo, y_hi, x_lo, x_hi) for the PSF footprint of *rec*."""
    xi = int(round(rec.x)); yi = int(round(rec.y))
    _hw = hw_override if hw_override is not None else hw
    y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, _hw, ny, nx)
    dx = rec.x - xi; dy = rec.y - yi
    local_psf = interpolate_psf(cube, xs, ys, rec.x + x_offset, rec.y + y_offset,
                               _cache=psf_cache)
    if not prefiltered:
        from scipy.ndimage import spline_filter
        coeffs = spline_filter(local_psf, order=3, output=np.float64)
    else:
        coeffs = local_psf
    P, _, _ = _eval_psf_grad_fast(coeffs, dx, dy, dix, diy, psf_scale)
    return P, y_lo, y_hi, x_lo, x_hi


def enforce_hmin(records, hmin):
    """Iteratively remove fainter stars within hmin of a brighter star."""
    if not records or hmin <= 0:
        return records

    from scipy.spatial import cKDTree
    sorted_recs = sorted(records, key=lambda r: r.flux, reverse=True)
    finite = [r for r in sorted_recs if np.isfinite(r.x) and np.isfinite(r.y)]
    if not finite:
        return []

    coords = np.array([[r.x, r.y] for r in finite])
    tree = cKDTree(coords)
    rejected = np.zeros(len(finite), dtype=bool)
    for i in range(len(finite)):
        if rejected[i]:
            continue
        for j in tree.query_ball_point(coords[i], hmin):
            if j > i:
                rejected[j] = True
    return [r for r, rej in zip(finite, rejected) if not rej]


def _fit_star_interp(data, x, y, psf_coeffs_cube, psf_xs, psf_ys, x_det, y_det,
                     psf_scale, hw, sky_init, gain, read_noise, **kwargs):
    """Interpolate PSF inside the worker thread, then fit. Avoids main-thread bottleneck."""
    psf_coeffs = interpolate_psf(psf_coeffs_cube, psf_xs, psf_ys, x_det, y_det)
    return fit_star(data, x, y, psf_coeffs, psf_scale, hw, sky_init, gain, read_noise, **kwargs)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def build_variance_image(records, psf_cube, xs, ys, psf_scale, shape,
                         gain, read_noise, x_offset=0.0, y_offset=0.0,
                         noise_map=None, psf_coeffs_cube=None, psf_cache=None,
                         gain_map=None, residual=None, mask=None,
                         read_noise_map=None):
    """Build a pixel-wise variance image accumulating Poisson noise from all star models."""
    ny, nx = shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None
    shw = _sub_hw(_cube, psf_scale)

    if noise_map is not None:
        # Start from the provided noise map (e.g., MIRI ERR^2)
        var_image = noise_map.astype(np.float64).copy()
    else:
        # Default Poisson base model
        good_sky = [r.sky for r in records
                    if _should_subtract(r) and np.isfinite(r.sky)]
        median_sky = float(np.median(good_sky)) if good_sky else 0.0
        
        # Scalar fallback for base variance
        var_base = max(median_sky, 0.0) / gain + (read_noise / gain) ** 2
        var_image = np.full((ny, nx), var_base, dtype=np.float64)
        
        # Apply spatial gain and read noise if available
        if gain_map is not None or read_noise_map is not None:
            g_eff = gain_map if gain_map is not None else gain
            rn_eff = read_noise_map if read_noise_map is not None else read_noise
            var_image = max(median_sky, 0.0) / g_eff + (rn_eff / g_eff) ** 2

    for rec in records:
        if not _should_subtract(rec):
            continue
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw=0, ny=ny, nx=nx,
            prefiltered=prefiltered, x_offset=x_offset, y_offset=y_offset,
            psf_cache=psf_cache, hw_override=shw)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue
        var_image[y_lo:y_hi, x_lo:x_hi] += np.maximum(rec.flux * P, 0.0) / gain

    return np.maximum(var_image, 1e-10)


def subtract_stars(residual, records, psf_cube, xs, ys, psf_scale, hw,
                   x_offset=0.0, y_offset=0.0,
                   psf_coeffs_cube=None, psf_cache=None):
    """Subtract PSF models of well-fit stars from *residual* in-place."""
    ny, nx = residual.shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None

    shw = _sub_hw(_cube, psf_scale)
    for rec in records:
        if not _should_subtract(rec):
            continue
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            prefiltered=prefiltered, x_offset=x_offset, y_offset=y_offset,
            psf_cache=psf_cache,
            hw_override=shw)
        if y_lo < y_hi and x_lo < x_hi:
            residual[y_lo:y_hi, x_lo:x_hi] -= rec.flux * P


def refit_stars(residual, records, psf_cube, xs, ys, psf_scale, hw,
                gain, read_noise, mask, noise_map, x_offset, y_offset,
                zero_point, exposure_time=1.0, max_iter=15, tol=1e-3, psf_coeffs_cube=None,
                sat_threshold=float('inf'), verbose=False, desc="Re-fitting",
                sigma_clip=True, sigma_clip_sigma=4.0, sigma_clip_iter=2,
                psf_cache=None, gain_map=None, read_noise_map=None):
    """Re-fit all stars in *records* via leave-one-out on *residual*."""
    ny, nx = residual.shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None

    iterator = (
        tqdm(range(len(records)), desc=desc, unit="star")
        if verbose else range(len(records))
    )

    shw = _sub_hw(_cube, psf_scale)

    for i in iterator:
        rec = records[i]
        was_subtracted = _should_subtract(rec)
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            prefiltered=prefiltered, x_offset=x_offset, y_offset=y_offset,
            psf_cache=psf_cache,
            hw_override=shw)
        if y_lo >= y_hi or x_lo >= x_hi:
            continue
        if was_subtracted:
            residual[y_lo:y_hi, x_lo:x_hi] += rec.flux * P

        new_rec = fit_star(
            residual, rec.x, rec.y, 
            _cube, psf_scale, hw, rec.sky, gain, read_noise,
            max_iter=max_iter, tol=tol, noise_map=noise_map, mask=mask,
            zero_point=zero_point,  pass_number=rec.pass_number,
            sigma_clip=sigma_clip,
            sigma_clip_sigma=sigma_clip_sigma,
            sigma_clip_iter=sigma_clip_iter,
            gain_map=gain_map,
            read_noise_map=read_noise_map, flux_unit_conversion=flux_unit_conversion
        )

        new_rec.n_sat = rec.n_sat

        P2, y_lo2, y_hi2, x_lo2, x_hi2 = _psf_window(
            new_rec, _cube, xs, ys, psf_scale, hw, ny, nx,
            prefiltered=prefiltered, x_offset=x_offset, y_offset=y_offset,
            psf_cache=psf_cache,
            hw_override=shw)
        if y_lo2 < y_hi2 and x_lo2 < x_hi2 and _should_subtract(new_rec):
            residual[y_lo2:y_hi2, x_lo2:x_hi2] -= new_rec.flux * P2

        records[i] = new_rec

    return records


def inject_stars(image, input_stars, psf_cube, xs, ys, psf_scale, hw, 
                 psf_coeffs_cube=None, zero_point=0.0, gain=1.0, 
                 exposure_time=1.0, x_offset=0.0, y_offset=0.0, add_noise=True):
    """
    Inject artificial stars from input_stars (X, Y, Mag) into image.
    Follows Jay Anderson's jwst1pass logic for evaluative ART tests.
    """
    ny, nx = image.shape
    _cube = psf_coeffs_cube if psf_coeffs_cube is not None else psf_cube
    prefiltered = psf_coeffs_cube is not None
    
    # Simple object to satisfy _psf_window expectations
    class MockRecord:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    shw = _sub_hw(_cube, psf_scale)
    
    for s in input_stars:
        if len(s) < 3: continue
        x, y, m = s[:3]
        # Direct translation of Fortran: fr = 10**(-mr/2.5)
        # Accounting for Python's normalized mag: mr = m - zp - 2.5*log10(exp)
        # Resulting in total DN flux: f = exp * 10**((zp - m)/2.5)
        flux = exposure_time * 10**((zero_point - m) / 2.5)
        rec = MockRecord(x=x, y=y)
        
        # We use a 25x25 window (hw=12) for injection to match Fortran
        P, y_lo, y_hi, x_lo, x_hi = _psf_window(
            rec, _cube, xs, ys, psf_scale, hw=12, ny=ny, nx=nx,
            prefiltered=prefiltered, x_offset=x_offset, y_offset=y_offset,
            hw_override=max(shw, 12))
            
        if y_lo < y_hi and x_lo < x_hi:
            added = flux * P
            image[y_lo:y_hi, x_lo:x_hi] += added
            
            if add_noise:
                # Jay's noise model: fsig = sqrt(fadd)
                # Since image is in DN, this matches gain=1.0 logic.
                noise = np.random.normal(0, np.sqrt(np.maximum(added, 0)))
                image[y_lo:y_hi, x_lo:x_hi] += noise


def run_photometry(data, psf_cube, xs, ys, psf_scale, mask=None, 
                   n_passes=2, n_discovery_passes=None, hmin=5.0, fmin=1000.0, hw=5,
                   sky_inner=4, sky_outer=8,
                   gain=1.0, read_noise=10.0, zero_point=0.0, exposure_time=1.0, 
                   x_offset=0.0, y_offset=0.0,
                   max_iter=15, tol=1e-3,
                   sat_threshold=float('inf'),
                   sigma_clip=True, sigma_clip_sigma=4.0,
                   n_jobs=-1, verbose=False,
                   pipeline_noise_map=None,
                   gain_map=None, read_noise_map=None,
                   input_stars=None,
                   inject=False, flux_unit_conversion=1.0,
                   conc_limit=0.9):
    """Run multi-pass photometry on a JWST image."""
    ny, nx = data.shape
    
    # Base data for fitting (includes injected stars if requested)
    working_data = data.astype(np.float64).copy()
    
    if n_discovery_passes is None:
        if input_stars is not None:
            n_discovery_passes = 0 # Skip discovery if input catalog provided
        else:
            n_discovery_passes = n_passes - 1 if n_passes > 1 else n_passes
    
    from scipy.ndimage import spline_filter
    psf_coeffs_cube = np.array([spline_filter(p, order=3, output=np.float64) for p in psf_cube])
    psf_cache: dict = {}
    
    # Use the pipeline noise map (ERR^2) as the starting point if provided (e.g., MIRI)
    _current_noise_map = pipeline_noise_map

    if verbose:
        if _current_noise_map is not None:
            print(f"  Noise model   : Pipeline ERR model (σ² = ERR²)")
        else:
            print(f"  Noise model   : Poisson model  (σ² = (model+sky)/gain + (RN/gain)²)")

        print(f"  Gain (noise)  : {gain:.4f}  "
              f"{'[target DN]' if gain == 1.0 else '[e-/DN]'}")
        print(f"  Read noise    : {read_noise:.2f} e-")
        print(f"  Image size    : {nx} × {ny} px")
        print(f"  PSF grid      : {psf_cube.shape[0]} PSFs  "
              f"(scale={psf_scale}×,  array={psf_cube.shape[-1]}×{psf_cube.shape[-2]} px)")
        print(f"  Fit window    : {2*hw+1}×{2*hw+1} px  (hw={hw})")
        print(f"  Sigma clip    : {'on' if sigma_clip else 'off'}"
              f"  (σ={sigma_clip_sigma}, iter=2)")

    # 0. Artificial Star Injection
    if input_stars is not None and inject:
        if verbose:
            print(f"Injecting {len(input_stars)} artificial stars into image...")
        inject_stars(working_data, input_stars, psf_cube, xs, ys, psf_scale, hw,
                     psf_coeffs_cube=psf_coeffs_cube, zero_point=zero_point, gain=gain,
                     
                     x_offset=x_offset, y_offset=y_offset)

    residual = working_data.copy()
    all_records = []
    
    # 1. Initial pass for input stars
    if input_stars is not None:
        n_input = len(input_stars)
        if verbose:
            print(f"Fitting {n_input} stars from input catalog...")
        
        # Initial sky estimation if none provided
        sky_global = estimate_sky_global(working_data, mask=mask)
        
        # Parallel fit of input stars
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_fit_star_interp)(
                working_data, s[0], s[1],
                psf_coeffs_cube, xs, ys, s[0] + x_offset, s[1] + y_offset,
                psf_scale, hw, sky_global,
                gain, read_noise, mask=mask,
                noise_map=_current_noise_map,
                max_iter=max_iter, tol=tol,
                zero_point=zero_point,
                pass_number=1,
                gain_map=gain_map,
                read_noise_map=read_noise_map, flux_unit_conversion=flux_unit_conversion
            ) for s in input_stars
        )
        
        all_records = [r for r in results if r is not None]
        
        if verbose:
            n_conv = sum(r.converged for r in all_records)
            print(f"  Fitted {len(all_records)} stars.")
            print(f"  Converged: {n_conv}/{len(all_records)} ({100*n_conv/max(len(all_records),1):.1f}%)")

        subtract_stars(residual, all_records, psf_cube, xs, ys, psf_scale, hw,
                       psf_coeffs_cube=psf_coeffs_cube, x_offset=x_offset, y_offset=y_offset,
                       psf_cache=psf_cache)

        _current_noise_map = build_variance_image(
            all_records, psf_cube, xs, ys, psf_scale, working_data.shape,
            gain, read_noise, x_offset, y_offset,
            noise_map=pipeline_noise_map,
            psf_coeffs_cube=psf_coeffs_cube,
            gain_map=gain_map,
            residual=residual,
            mask=mask,
            read_noise_map=read_noise_map,
            psf_cache=psf_cache)

    for p in range(1, n_passes + 1):
        do_discovery = p <= n_discovery_passes
        if verbose:
            print(f"Pass {p}/{n_passes}...")
        
        if do_discovery:
            # 1. Source Discovery
            if verbose:
                print(f"  Finding sources (fmin={fmin})...")
            # We detect on the RESIDUAL of previous pass
            xs_c, ys_c, peaks_c, skys_c, sigs_c = find_sources(
                residual, sky_inner, sky_outer, hmin, fmin, mask=mask,
                suppress_radius=2.0)
            
            n_cand = len(xs_c)
            if n_cand > 0:
                if verbose:
                    print(f"  Found {n_cand} candidates. Fitting...")
                
                # 2. Iterative Fitting
                results = Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(_fit_star_interp)(
                        working_data, xs_c[i], ys_c[i],
                        psf_coeffs_cube, xs, ys, xs_c[i] + x_offset, ys_c[i] + y_offset,
                        psf_scale, hw, skys_c[i],
                        gain, read_noise, mask=mask,
                        noise_map=_current_noise_map,
                        max_iter=max_iter, tol=tol,
                        zero_point=zero_point,
                        pass_number=p,
                        gain_map=gain_map,
                        read_noise_map=read_noise_map, flux_unit_conversion=flux_unit_conversion
                    ) for i in range(n_cand)
                )
                
                pass_records = [r for r in results if r is not None]
                
                # 3. Enforce hmin and Update Residual
                all_records = enforce_hmin(all_records + pass_records, hmin)
                
                # Re-calculate residual from scratch
                residual = working_data.copy()
                subtract_stars(residual, all_records, psf_cube, xs, ys, psf_scale, hw,
                               psf_coeffs_cube=psf_coeffs_cube, x_offset=x_offset, y_offset=y_offset,
                               psf_cache=psf_cache)
                
                if verbose:
                    curr_pass_kept = [r for r in all_records if r.pass_number == p]
                    n_conv = sum(r.converged for r in curr_pass_kept)
                    print(f"  Fitted {len(pass_records)} new stars.")
                    print(f"  Converged (new): {n_conv}/{max(len(curr_pass_kept),1)} ({100*n_conv/max(len(curr_pass_kept),1):.1f}%)")
            else:
                if verbose:
                    print("  No new candidates found.")
                break
        
        # 4. Global Refinement
        if p < n_passes or (do_discovery and n_cand > 0):
            n_all = len(all_records)
            if verbose:
                print(f"  Refining all {n_all} stars...")
            
            results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_fit_star_interp)(
                    working_data, r.x, r.y,
                    psf_coeffs_cube, xs, ys, r.x + x_offset, r.y + y_offset,
                    psf_scale, hw, r.sky,
                    gain, read_noise, mask=mask,
                    noise_map=_current_noise_map,
                    max_iter=max_iter, tol=tol,
                    zero_point=zero_point,
                    pass_number=p,
                    gain_map=gain_map,
                    read_noise_map=read_noise_map, flux_unit_conversion=flux_unit_conversion
                ) for r in all_records
            )
            
            all_records = [r for r in results if r is not None]
            
            if verbose:
                n_conv = sum(r.converged for r in all_records)
                print(f"  Refitted {len(all_records)} stars.")
                print(f"  Converged (total): {n_conv}/{max(len(all_records),1)} ({100*n_conv/max(len(all_records),1):.1f}%)")

        # 5. Noise Model Update
        _current_noise_map = build_variance_image(
            all_records, psf_cube, xs, ys, psf_scale, working_data.shape,
            gain, read_noise, x_offset, y_offset,
            noise_map=pipeline_noise_map,
            psf_coeffs_cube=psf_coeffs_cube,
            gain_map=gain_map,
            residual=residual,
            mask=mask,
            read_noise_map=read_noise_map,
            psf_cache=psf_cache)
    # Final step: Final hmin filter and Error Inflation
    all_records = enforce_hmin(all_records, hmin)
    
    if verbose:
        print("Finalizing errors...")
    inflate_chi2(all_records, zero_point=zero_point, exposure_time=exposure_time)
    compute_neighbor_stats(all_records, hw)
    classify_stars(all_records, conc_lo=conc_limit)

    return all_records, residual, _current_noise_map
