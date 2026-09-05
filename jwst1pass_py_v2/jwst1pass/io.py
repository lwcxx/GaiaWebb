"""FITS image loader and PSF loader for JWST (MIRI/NIRISS/NIRCam)."""

import os
import urllib.request
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# Instrument/detector → prefix used in STDPSF/STDGDC filenames
_DETECTOR_PREFIX = {
    ('MIRI',   'MIRIM'): 'MIRI',
    ('NIRISS', 'NIS'):   'NIRISS',
    # NIRCam single-chip detectors — prefix matches actual filename (STDPSF_NRCA1_F115W.fits)
    ('NIRCAM', 'NRCA1'): 'NRCA1',
    ('NIRCAM', 'NRCA2'): 'NRCA2',
    ('NIRCAM', 'NRCA3'): 'NRCA3',
    ('NIRCAM', 'NRCA4'): 'NRCA4',
    ('NIRCAM', 'NRCB1'): 'NRCB1',
    ('NIRCAM', 'NRCB2'): 'NRCB2',
    ('NIRCAM', 'NRCB3'): 'NRCB3',
    ('NIRCAM', 'NRCB4'): 'NRCB4',
    ('NIRCAM', 'NRCALONG'): 'NRCAL',
    ('NIRCAM', 'NRCBLONG'): 'NRCBL',
}

# NIRCam SWC detectors live under lib/STDPSFs/NIRCam/SWC/{filter}/
_NIRCAM_SWC_DETECTORS = {'NRCA1','NRCA2','NRCA3','NRCA4','NRCB1','NRCB2','NRCB3','NRCB4'}
_NIRCAM_LWC_DETECTORS = {'NRCALONG','NRCBLONG'}

# NIRCam detector name → internal instrument code (INSTu in Fortran)
# Matches Fortran jwst1pass lines 14342-14349
_NIRCAM_DETECTOR_INST = {
    'NRCA1': 21, 'NRCA2': 22, 'NRCA3': 23, 'NRCA4': 24,
    'NRCB1': 25, 'NRCB2': 26, 'NRCB3': 27, 'NRCB4': 28,
}

# Chip boundaries for the NIRCam 8-chip meta mosaic (8192×4096).
# Each tuple: (col_lo, col_hi, row_lo, row_hi) — 0-based Python (data[row, col]).
# Translated from Fortran BDRY_XR/YR (1-based, X=column, Y=row):
#   Chip 1: col 1-2048,   row 1-2048
#   Chip 2: col 1-2048,   row 2049-4096
#   Chip 3: col 2049-4096, row 1-2048
#   Chip 4: col 2049-4096, row 2049-4096
#   Chip 5: col 6145-8192, row 2049-4096
#   Chip 6: col 6145-8192, row 1-2048
#   Chip 7: col 4097-6144, row 2049-4096
#   Chip 8: col 4097-6144, row 1-2048
_NIRCAM_META_CHIP_BOUNDS = [
    (0,    2048, 0,    2048),  # chip 1
    (0,    2048, 2048, 4096),  # chip 2
    (2048, 4096, 0,    2048),  # chip 3
    (2048, 4096, 2048, 4096),  # chip 4
    (6144, 8192, 2048, 4096),  # chip 5
    (6144, 8192, 0,    2048),  # chip 6
    (4096, 6144, 2048, 4096),  # chip 7
    (4096, 6144, 0,    2048),  # chip 8
]

# Flag values for NIRCam (Fortran lines 14338-14339, 14624-14625)
_NIRCAM_LOFLAG = -100.0
_NIRCAM_HIFLAG = 60000.0
_NIRCAM_META_GAP_VAL = -750.0  # stamped into inter-chip gap columns/rows

def _extract_filter(header, instrume):
    """Extract the science filter name from a JWST header."""
    instrume = instrume.strip().upper()
    filt = header.get('FILTER', '').strip().upper()
    pupil = header.get('PUPIL', '').strip().upper()
    if instrume == 'NIRISS' and filt == 'CLEAR':
        return pupil
    # NIRCam uses FILTER directly; PUPIL may contain the short-wavelength filter
    # when the long-wavelength filter is in FILTER.
    return filt


def get_chip_config(instrume, detector):
    """Return per-chip configuration for JWST instruments.

    Returns a list of (sci_ext, dq_ext, y_offset) tuples.
    For NIRCam meta mosaics the multi-chip loop is handled entirely inside
    run_photometry_fits, so this function returns a single dummy entry.
    """
    instrume = instrume.strip().upper()
    if instrume == 'NIRCAM':
        return [(1, None, 0.0)]
    return [(1, None, 0.0)]  # dq_ext=None → load_image finds DQ by name (ext 3)


# ---------------------------------------------------------------------------
# NIRCam image loading helpers
# (Translation of Fortran jwst1pass lines 14319-15068)
# ---------------------------------------------------------------------------

def _mask_nircam_meta_gaps(data, gap_val=_NIRCAM_META_GAP_VAL):
    """Stamp gap_val into the inter-chip gap columns and row of a NIRCam 8192×4096 mosaic.

    Direct translation of Fortran jwst1pass lines 14639-14700.
    Fortran uses pixc(col, row) 1-based; here data[row, col] 0-based.
    """
    # Row gap at the A/B module boundary (Fortran j=2044..2053, 1-based → rows 2043..2052)
    data[2043:2053, :] = gap_val

    # Column gaps (Fortran i values, 1-based → 0-based col indices)
    data[:, 0:5]       = gap_val   # i=1..5
    data[:, 2043:2053] = gap_val   # i=2044..2053
    data[:, 4091:4101] = gap_val   # i=4092..4101
    data[:, 6139:6148] = gap_val   # i=6140..6148
    data[:, 8187:8192] = gap_val   # i=8188..8192


def _interpolate_saturated_pixels(data, sat_mask, hiflag, max_radius=9, border=15):
    """Iteratively fill saturated pixels with distance-weighted averages of neighbours.

    Direct translation of Fortran jwst1pass lines 14401-14446.

    For each saturated pixel the aperture radius is expanded until fewer than half
    of the pixels inside it are also saturated (or the radius exceeds max_radius).
    Each unsaturated neighbour contributes pixel_value / (1 + distance) to a
    weighted sum that replaces the saturated pixel.  Saturated neighbours
    contribute hiflag / (1 + distance) to the weight tally.

    Processes in raster order so a pixel filled earlier can feed later ones,
    mirroring the Fortran loop structure.
    """
    result = data.copy()
    ny, nx = data.shape

    sat_ys, sat_xs = np.where(sat_mask)
    valid = ((sat_ys >= border) & (sat_ys < ny - border) &
             (sat_xs >= border) & (sat_xs < nx - border))
    sat_ys = sat_ys[valid]
    sat_xs = sat_xs[valid]

    # Sort in raster order (top-left to bottom-right) to match Fortran loop ordering
    order = np.lexsort((sat_xs, sat_ys))
    sat_ys = sat_ys[order]
    sat_xs = sat_xs[order]

    for iy, ix in zip(sat_ys, sat_xs):
        ir = 0
        while True:
            ir += 1
            tn = ts = 0
            tp = 0.0

            for diy in range(-ir, ir + 1):
                for dix in range(-ir, ir + 1):
                    rr = float(np.sqrt(diy * diy + dix * dix))
                    if rr <= ir + 0.5:
                        tn += 1
                        jy, jx = iy + diy, ix + dix
                        if 0 <= jy < ny and 0 <= jx < nx:
                            if sat_mask[jy, jx]:
                                pu = float(hiflag)
                                ts += 1
                            else:
                                pu = float(result[jy, jx])
                            tp += pu / (1.0 + rr)

            if ts < tn / 2 or ir >= max_radius:
                break

        result[iy, ix] = tp

    return result


def load_nircam_single(image_path, verbose=False, lib_dir=None):
    """Load a single NIRCam 2048×2048 detector _cal.fits image.

    Translation of Fortran jwst1pass lines 14332-14609.

    Performs NIRCam-specific preprocessing:
    - Dead pixel detection via DQ bitmask (bits 0-1 not set + high DQ value)
    - Saturation detection (pixc==0 AND DQ bits 0-1 both set)
    - Iterative saturation interpolation (expanding circle, max radius 9)
    - Clamp of any pixel below -900 to exactly -900

    Returns the same 10-tuple as load_image:
        (data, effective_gain, effective_rn, mask, header,
         x_offset, y_offset, pipeline_noise_map, gain_map, read_noise_map)
    """
    LOFLAG = _NIRCAM_LOFLAG
    HIFLAG = _NIRCAM_HIFLAG

    with fits.open(image_path) as hdul:
        primary_hdr = hdul[0].header

        # Extension 1: SCI (float32)
        sci_hdr = hdul[1].header
        data = hdul[1].data.astype(np.float64)

        # Extension 2: ERR (float32) — search by name for robustness
        err_data = None
        for i in range(1, len(hdul)):
            if 'ERR' in hdul[i].name.upper():
                err_data = hdul[i].data.astype(np.float64)
                break

        # Extension 3: DQ (int32)
        dq_data = None
        for i in range(1, len(hdul)):
            if 'DQ' in hdul[i].name.upper():
                try:
                    dq_data = hdul[i].data.astype(np.int32)
                except (ValueError, TypeError):
                    pass
                break

        header = primary_hdr.copy()
        header.update(sci_hdr)

    # NaN → 0 (Fortran line 14363: if (isnan(pixc)) pixc = 0.0)
    data[~np.isfinite(data)] = 0.0

    # Dead pixel masking (Fortran lines 14368-14383):
    # if iand(pixn, 3) != 3  AND  pixn > 1e6  AND  pixc <= 0.00001
    #   → stamp 3×3 neighbourhood with LOFLAG-1
    if dq_data is not None:
        dead = (((dq_data & 3) != 3) &
                (dq_data > int(1e6)) &
                (data <= 0.00001))
        dead_ys, dead_xs = np.where(dead)
        ny, nx = data.shape
        for iy, ix in zip(dead_ys, dead_xs):
            if 1 <= iy < ny - 1 and 1 <= ix < nx - 1:
                data[iy - 1:iy + 2, ix - 1:ix + 2] = LOFLAG - 1.0

    # Saturation mask (Fortran lines 14389-14393):
    # pixq=1 if pixc==0 AND iand(pixn, 3)==3
    if dq_data is not None:
        sat_mask = (data == 0.0) & ((dq_data & 3) == 3)
    else:
        sat_mask = data == 0.0

    nsat = int(sat_mask.sum())
    if verbose:
        print(f"  NIRCam single-chip NSAT: {nsat}")

    # Iterative saturation interpolation (Fortran lines 14401-14446)
    if nsat > 0:
        data = _interpolate_saturated_pixels(data, sat_mask, HIFLAG,
                                             max_radius=9, border=15)

    # Clamp extreme-negative pixels: if not (pixc > -900) → pixc = -900
    # (Fortran lines 14598-14601)
    data[~(data > -900.0)] = -900.0

    # Pipeline noise map from ERR extension (variance = ERR²)
    pipeline_noise_map = None
    if err_data is not None:
        pipeline_noise_map = err_data ** 2
        pipeline_noise_map[~np.isfinite(pipeline_noise_map)] = 1e9
        pipeline_noise_map[pipeline_noise_map <= 0] = 1e9

    # DQ-based bad-pixel mask for the fitter
    # Saturated pixels and pixels below LOFLAG are masked
    mask = sat_mask | (data <= LOFLAG)

    noise_info = _get_noise_info(primary_hdr, sci_hdr, 'NIRCAM')
    # Scale gain to match image units (MJy/sr → electrons conversion).
    # Poisson variance in (MJy/sr)² = flux_MJy_sr / g_eff, where
    # g_eff = hardware_gain × xfactor  [e- per MJy/sr].
    # Without xfactor the Poisson term is xfactor× too large, severely
    # down-weighting bright-star pixels and biasing sky estimates.
    bunit = sci_hdr.get('BUNIT', primary_hdr.get('BUNIT', '')).strip().upper()
    if bunit == 'MJY/SR':
        photmjsr = sci_hdr.get('PHOTMJSR', primary_hdr.get('PHOTMJSR', 1.0))
        effexptm = primary_hdr.get('EFFEXPTM', sci_hdr.get('EFFEXPTM', 1.0))
        xfactor = effexptm / max(photmjsr, 1e-30)
        effective_gain = noise_info['hardware_gain'] * xfactor
    else:
        effective_gain = noise_info['hardware_gain']
    effective_rn = noise_info['read_noise']

    gain_map = _get_gain_map('NIRCAM', data.shape, header=header, lib_dir=lib_dir)
    read_noise_map = _get_read_noise_map('NIRCAM', data.shape, header=header,
                                         lib_dir=lib_dir, gain_map=gain_map)

    # gain_map is in e-/DN (hardware units). For MJy/sr data the Poisson term
    # in the fitter is flux_MJysr * P / g_eff, which must be in (MJy/sr)².
    # Multiplying by xfactor converts g_eff to e-/(MJy/sr), matching
    # effective_gain and yielding Poisson_var / ERR²_pipeline ≈ 1 for
    # Poisson-limited pixels instead of the 306× inflation without this scaling.
    if bunit == 'MJY/SR' and gain_map is not None:
        gain_map = gain_map * xfactor

    return (data, effective_gain, effective_rn,
            mask, header, 0.0, 0.0, pipeline_noise_map, gain_map, read_noise_map)


def load_nircam_meta_full(image_path, verbose=False):
    """Load a NIRCam 8-chip meta mosaic (single 8192×4096 FITS image).

    Translation of Fortran jwst1pass lines 14618-15006.

    The meta format stores all 8 short-wavelength NIRCam detectors stitched into
    one image.  There is no ERR or DQ extension.  Saturation is indicated purely
    by zero pixel values.

    Returns:
        data      : np.ndarray (4096, 8192) — gap-masked, saturation-replaced mosaic
        full_mask : np.ndarray (4096, 8192) bool — bad-pixel mask
        header    : combined FITS header
    """
    with fits.open(image_path) as hdul:
        primary_hdr = hdul[0].header
        sci_hdr = hdul[1].header
        data = hdul[1].data.astype(np.float64)   # (4096, 8192)
        header = primary_hdr.copy()
        header.update(sci_hdr)

    if data.shape != (4096, 8192):
        raise ValueError(
            f"NIRCam meta mosaic: expected shape (4096, 8192), got {data.shape}")

    # Apply inter-chip gap masking (Fortran lines 14639-14700)
    _mask_nircam_meta_gaps(data, gap_val=_NIRCAM_META_GAP_VAL)

    # Saturation: pixc==0 → pixq=1 (Fortran lines 14693-14700)
    sat_mask = (data == 0.0)
    # Replace saturated pixels with HIFLAG+1 so the fitter skips them
    data[sat_mask] = _NIRCAM_HIFLAG + 1.0

    nsat = int(sat_mask.sum())
    if verbose:
        print(f"  NIRCam meta NSAT: {nsat}")

    # Combined bad-pixel mask: saturated OR gap
    gap_mask = data <= _NIRCAM_META_GAP_VAL
    full_mask = sat_mask | gap_mask

    return data, full_mask, header


def load_image(image_path, ext=1, dq_ext=None, dq_flags=None, lib_dir=None, verbose=False):
    """Load a JWST _cal image, apply unit conversion, and return noise/mask info.

    For NIRCam single-chip 2048×2048 images the call is forwarded to
    load_nircam_single(), which applies NIRCam-specific DQ masking and
    iterative saturation interpolation.
    """
    # Route NIRCam single-chip images to their dedicated loader
    with fits.open(image_path) as _hdul:
        _instrume = _hdul[0].header.get('INSTRUME', '').strip().upper()
        _naxis1 = _hdul[1].header.get('NAXIS1', 0) if len(_hdul) > 1 else 0
    if _instrume == 'NIRCAM' and _naxis1 == 2048:
        return load_nircam_single(image_path, verbose=verbose, lib_dir=lib_dir)

    with fits.open(image_path) as hdul:
        primary_hdr = hdul[0].header
        sci_hdr = hdul[ext].header
        data = hdul[ext].data.astype(np.float64)
        
        # Combine headers for metadata access
        header = primary_hdr.copy()
        header.update(sci_hdr)
        
        instrume = primary_hdr.get('INSTRUME', '').strip().upper()
        detector = primary_hdr.get('DETECTOR', '').strip().upper()

        # Noise and DQ
        noise_info = _get_noise_info(primary_hdr, sci_hdr, instrume)
        gain_map = _get_gain_map(instrume, data.shape, header=header, lib_dir=lib_dir)
        read_noise_map = _get_read_noise_map(instrume, data.shape, header=header, lib_dir=lib_dir, gain_map=gain_map)

        # Automatic DQ selection if not provided
        if dq_ext is None:
            for i in range(1, len(hdul)):
                if 'DQ' in hdul[i].name.upper():
                    dq_ext = i
                    break
        mask = _load_dq_mask(hdul, dq_ext, dq_flags)
        
        photmjsr = header.get('PHOTMJSR', 1.0)
        effexptm = header.get('EFFEXPTM', 1.0)
        xfactor = effexptm / photmjsr
        units = header.get('BUNIT', '').strip().upper()

        # All MJy/sr instruments (NIRISS, MIRI, NIRCam) stay in native MJy/sr units.
        # This matches the Fortran engine default (XFACTOR=1, no scaling applied),
        # so fmin=5 means 5 MJy/sr above sky for every camera consistently.

        # Load the ERR extension to provide a pipeline-based noise map
        pipeline_noise_map = None
        err_ext = None
        for i in range(1, len(hdul)):
            if 'ERR' in hdul[i].name.upper():
                err_ext = i
                break
        if err_ext is not None:
            err_data = hdul[err_ext].data.astype(np.float64)
            # Variance = ERR^2 in (MJy/sr)^2 — no unit conversion needed
            pipeline_noise_map = (err_data**2)

            # Mask invalid/negative variances
            pipeline_noise_map[~np.isfinite(pipeline_noise_map)] = 1e9
            pipeline_noise_map[pipeline_noise_map <= 0] = 1e9

        x_offset, y_offset = _detector_offsets(instrume, detector, ext)

        # Scale noise constants to match image units
        # Effective gain is electrons per image unit (MJy/sr or DN)
        hardware_gain = noise_info['hardware_gain']
        rn_dn = noise_info['read_noise']

        if units == 'MJY/SR':
            # e- per MJy/sr: hardware_gain [e-/DN] × xfactor [DN per MJy/sr]
            effective_gain = hardware_gain * xfactor
        else: # Default (usually electrons)
            effective_gain = 1.0 if units == 'ELECTRONS' else hardware_gain

        # RN passed to fit_star should be in electrons
        effective_rn = rn_dn * hardware_gain

        # Scale spatial gain map from e-/DN to e-/(MJy/sr) for MJy/sr data
        if gain_map is not None and units == 'MJY/SR':
            gain_map *= xfactor

        return (data, effective_gain, effective_rn,
                mask, header, x_offset, y_offset, pipeline_noise_map, gain_map, read_noise_map)


def _load_dq_mask(hdul, dq_ext, dq_flags):
    """Load DQ mask from a FITS extension."""
    if dq_ext is None:
        return None
    try:
        dq_data = hdul[dq_ext].data
        # Handle NaNs in DQ if they exist (unlikely but safe)
        if np.issubdtype(dq_data.dtype, np.floating):
            dq = np.nan_to_num(dq_data, nan=0).astype(np.int32)
        else:
            dq = np.array(dq_data, dtype=np.int32)
            
        if dq_flags is None:
            return dq != 0
        mask = np.zeros(dq.shape, dtype=bool)
        for f in dq_flags:
            mask |= (dq & int(f)) != 0
        return mask
    except (IndexError, KeyError):
        return None


def _get_noise_info(primary_hdr, sci_hdr, instrume):
    """Extract noise parameters from JWST headers."""
    instrume = instrume.strip().upper()
    
    # Logic synced with jwst1pass (Fortran) around line 23679 (LC version):
    # NIRISS uses empirical gains for 4 amplifiers:
    # Amp A: 2.020  (ATODGNA)
    # Amp B: 1.886  (ATODGNB)
    # Amp C: 2.017  (ATODGNC)
    # Amp D: 2.011  (ATODGND)
    # Mean gain = 1.9835
    # The noise model in the fit loop uses an effective read noise floor 
    # of 5.0 DN (variance 25.0) and assumes data is in DN (Effective Gain=1.0).
    
    if instrume == 'NIRISS':
        gain_hdr = 1.9835 # Mean of calibrated amplifier gains A,B,C,D
        rn_hdr = 5.0      # 5.0 DN effective read noise
    elif instrume == 'MIRI':
        # MIRI pipeline images already account for detector gain, so hardware_gain=1.0.
        # effective_gain = hardware_gain * xfactor = xfactor [e- per MJy/sr].
        gain_hdr = 1.0
        rn_hdr = 5.0      # Fallback RN if ERR is missing
    elif instrume == 'NIRCAM':
        # NIRCam HgCdTe detector, similar noise model to NIRISS.
        # Images stay in MJy/sr (no XFACTOR conversion applied).
        gain_hdr = 2.05   # Typical NIRCam amplifier gain (e-/DN)
        rn_hdr = 15.0     # Typical NIRCam effective read noise (electrons)
    else:
        gain_hdr = sci_hdr.get('GAIN', primary_hdr.get('GAIN', 1.0))
        rn_hdr = sci_hdr.get('READNOIS', primary_hdr.get('READNOIS', 5.0))
    
    bunit = (sci_hdr.get('BUNIT', '')).strip().upper()
    
    return {
        'hardware_gain':    float(gain_hdr),
        'effective_gain':   1.0,  # Target DN units for photometry loop
        'read_noise':       float(rn_hdr),
        'bunit':            bunit,
        'gain_from_header': 'GAIN' in sci_hdr or 'GAIN' in primary_hdr,
        'rn_from_header':   'READNOIS' in sci_hdr or 'READNOIS' in primary_hdr,
        'data_in_electrons': bunit == 'ELECTRONS',
    }


def _fetch_crds_reference(ref_name, target_dir, verbose=True):
    """Download a JWST CRDS reference file if not already present locally.

    Downloads from https://jwst-crds.stsci.edu/unchecked_get/references/jwst/<ref_name>
    and saves to <target_dir>/<ref_name>.  Uses a .downloading temp suffix to
    prevent partially-written files from being mistaken for complete ones.

    Returns the local path on success, or None if the download fails.
    """
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, ref_name)
    if os.path.exists(dest):
        return dest
    url = f"https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{ref_name}"
    tmp = dest + '.downloading'
    if verbose:
        print(f"  [CRDS] Downloading {ref_name} → {dest}")
    try:
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)
        if verbose:
            size_mb = os.path.getsize(dest) / 1e6
            print(f"  [CRDS] Saved {ref_name} ({size_mb:.1f} MB)")
        return dest
    except Exception as exc:
        if verbose:
            print(f"  [CRDS] WARNING: could not download {ref_name}: {exc}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


def _get_gain_map(instrume, shape, header=None, lib_dir=None):
    """Return a 2D spatial gain map for the instrument, prioritizing reference files."""
    ny, nx = shape
    
    # 1. Try to load from reference file if provided
    if header is not None and lib_dir is not None:
        r_gain = header.get('R_GAIN', '')
        if r_gain and r_gain.startswith('crds://'):
            ref_name = r_gain.replace('crds://', '')
            ref_dir = os.path.join(lib_dir, 'R_GAIN')
            ref_path = os.path.join(ref_dir, ref_name)
            if not os.path.exists(ref_path):
                fetched = _fetch_crds_reference(ref_name, ref_dir)
                if fetched:
                    ref_path = fetched
            if os.path.exists(ref_path):
                try:
                    with fits.open(ref_path) as hdul:
                        # Reference maps are usually in EXT 1 (SCI)
                        gmap = hdul[1].data.astype(np.float64)
                        if gmap.shape == (ny, nx):
                            return gmap
                except Exception:
                    pass

    # 2. Fallback to hardcoded NIRISS model
    if instrume == 'NIRISS':
        # 4 amplifiers, 512 columns each
        # Amp C (2.017), Amp D (2.011), Amp A (2.020), Amp B (1.886)
        gmap = np.ones((ny, nx), dtype=np.float64)
        gmap[:, 0:512] = 2.017
        gmap[:, 512:1024] = 2.011
        gmap[:, 1024:1536] = 2.020
        gmap[:, 1536:2048] = 1.886
        return gmap
    return None


def _get_read_noise_map(instrume, shape, header=None, lib_dir=None, gain_map=None):
    """Return a 2D spatial read noise map (in electrons)."""
    ny, nx = shape

    # 1. Try to load from reference file
    if header is not None and lib_dir is not None:
        r_rn = header.get('R_READNO', '')
        if r_rn and r_rn.startswith('crds://'):
            ref_name = r_rn.replace('crds://', '')
            ref_dir = os.path.join(lib_dir, 'R_READNO')
            ref_path = os.path.join(ref_dir, ref_name)
            if not os.path.exists(ref_path):
                fetched = _fetch_crds_reference(ref_name, ref_dir)
                if fetched:
                    ref_path = fetched
            if os.path.exists(ref_path):
                try:
                    with fits.open(ref_path) as hdul:
                        rn_map = hdul[1].data.astype(np.float64)
                        if rn_map.shape == (ny, nx):
                            # Reference RN maps are in DN; convert to electrons.
                            # MIRI files (e.g. 0085) have blank BUNIT but are still DN.
                            bunit = hdul[1].header.get('BUNIT', '').strip().upper()
                            needs_conversion = (bunit == 'DN') or (bunit == '' and instrume == 'MIRI')
                            if needs_conversion:
                                if gain_map is not None:
                                    return rn_map * gain_map
                                # Instrument-specific scalar gain fallbacks
                                elif instrume == 'NIRISS':
                                    return rn_map * 1.9835
                                elif instrume == 'NIRCAM':
                                    return rn_map * 2.05
                                elif instrume == 'MIRI':
                                    return rn_map * 4.5
                            return rn_map
                except Exception:
                    pass
    return None


def _get_noise_params(hdr, instrume):
    """Compatibility wrapper."""
    info = _get_noise_info(hdr, hdr, instrume)
    return info['effective_gain'], info['read_noise']


def _detector_offsets(instrume, detector, sci_ext):
    """Return detector offsets (stub for JWST)."""
    return 0.0, 0.0


def _sm_suffix(mjd_obs, instrume, detector):
    """Return PSF filename SM-era suffix (stub for JWST)."""
    return ''


def _read_psf_positions(hdr, prefix, n):
    """Read IPSFX01..IPSFXnn or JPSFY01..JPSFYnn keywords."""
    vals = []
    for i in range(1, n + 1):
        for fmt in (f'{prefix}{i:02d}', f'{prefix}{i:04d}'):
            if fmt in hdr:
                vals.append(float(hdr[fmt]))
                break
    return vals


def load_stdpsf(path):
    """Alias for load_psf_cube."""
    return load_psf_cube(path)


def load_psf_cube(path):
    """Load a STDPSF FITS file and return the PSF array and grid info."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        psf_raw = hdul[0].data.astype(np.float64)
        
    if psf_raw.ndim == 2:
        psf_raw = psf_raw[np.newaxis]
        
    n_psf = psf_raw.shape[0]
    psf_scale = int(hdr.get('PSFSCALE', hdr.get('OVERSAMP', 4)))
    
    nx_g = int(hdr.get('NXPSFS', 1))
    ny_g = int(hdr.get('NYPSFS', 1))
    
    # Read grid positions from header
    xs = _read_psf_positions(hdr, 'IPSFX', nx_g)
    if not xs:
        xs = np.linspace(0, 2048, nx_g)
            
    ys = _read_psf_positions(hdr, 'JPSFY', ny_g)
    if not ys:
        ys = np.linspace(0, 2048, ny_g)
            
    return psf_raw, np.array(xs), np.array(ys), psf_scale, (ny_g, nx_g)


def find_psf(psf_dir, header):
    """Find the best-matching STDPSF file for a JWST header."""
    instrume = header.get('INSTRUME', '').strip().upper()
    detector = header.get('DETECTOR', '').strip().upper()
    filt = _extract_filter(header, instrume)

    prefix = _DETECTOR_PREFIX.get((instrume, detector), instrume)

    # Standard naming: STDPSF_{PREFIX}_{FILTER}.fits
    filename = f'STDPSF_{prefix}_{filt}.fits'

    candidates = []

    # NIRCam: lib uses NIRCam/SWC/{filter}/ or NIRCam/LWC/{filter}/ layout
    if instrume == 'NIRCAM':
        if detector in _NIRCAM_SWC_DETECTORS:
            channel = 'SWC'
        else:
            channel = 'LWC'
        candidates += [
            os.path.join(psf_dir, channel, filt, filename),
            os.path.join(psf_dir, channel, filename),
        ]

    # Generic fallback layouts
    candidates += [
        os.path.join(psf_dir, prefix, filename),
        os.path.join(psf_dir, filename),
        os.path.join(psf_dir, 'STDPSFs', prefix, filename),
        os.path.join(psf_dir, 'STDPSFs', filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No PSF found for {instrume}/{detector} {filt} in {psf_dir}.\n"
        f"Searched paths: {candidates}")


def find_gdc(gdc_dir, header):
    """Find the best-matching STDGDC file for a JWST header."""
    instrume = header.get('INSTRUME', '').strip().upper()
    detector = header.get('DETECTOR', '').strip().upper()
    filt = _extract_filter(header, instrume)

    prefix = _DETECTOR_PREFIX.get((instrume, detector), instrume)

    filename = f'STDGDC_{prefix}_{filt}.fits'

    candidates = []

    # NIRCam: lib uses NIRCam/SWC/{filter}/ or NIRCam/LWC/{filter}/ layout
    if instrume == 'NIRCAM':
        channel = 'SWC' if detector in _NIRCAM_SWC_DETECTORS else 'LWC'
        candidates += [
            os.path.join(gdc_dir, channel, filt, filename),
            os.path.join(gdc_dir, channel, filename),
        ]

    candidates += [
        os.path.join(gdc_dir, prefix, filename),
        os.path.join(gdc_dir, filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def load_stdgdc(path):
    """Load a STDGDC FITS file."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        xgc = np.array(hdul[1].data, dtype=np.float64)
        ygc = np.array(hdul[2].data, dtype=np.float64)
        mgc = np.array(hdul[3].data, dtype=np.float64)
        
    return {
        'xgc': xgc, 'ygc': ygc, 'mgc': mgc,
        'nx': int(hdr['NDIM_XGC']),
        'ny': int(hdr['NDIM_YGC'])
    }


def apply_gdc(x_raw, y_raw, gdc):
    """Apply forward GDC and magnitude correction (bilinear interpolation)."""
    nx, ny = gdc['nx'], gdc['ny']
    x = np.atleast_1d(np.asarray(x_raw, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y_raw, dtype=np.float64))
    
    # Boundary validation
    valid = (x >= -1.0) & (x <= nx + 1.0) & (y >= -1.0) & (y <= ny + 1.0)

    ix = np.clip(np.floor(x).astype(int), 0, nx - 2)
    iy = np.clip(np.floor(y).astype(int), 0, ny - 2)
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    
    def _bilin(arr):
        return ((1 - fx) * (1 - fy) * arr[iy, ix] +
                (1 - fx) * fy * arr[iy + 1, ix] +
                fx * (1 - fy) * arr[iy, ix + 1] +
                fx * fy * arr[iy + 1, ix + 1])
                
    ux = np.where(valid, _bilin(gdc['xgc']), np.nan)
    uy = np.where(valid, _bilin(gdc['ygc']), np.nan)
    mc = np.where(valid, _bilin(gdc['mgc']), 0.0)

    if np.asarray(x_raw).ndim == 0:
        return ux[0], uy[0], mc[0]
    return ux, uy, mc


def _gdc_jacobian_batch(x_raw, y_raw, gdc, step=0.001):
    """Compute GDC Jacobian d(x_gdc,y_gdc)/d(x,y) via central differences."""
    xp, yp, _ = apply_gdc(x_raw + step, y_raw, gdc)
    xm, ym, _ = apply_gdc(x_raw - step, y_raw, gdc)
    xu, yu, _ = apply_gdc(x_raw, y_raw + step, gdc)
    xd, yd, _ = apply_gdc(x_raw, y_raw - step, gdc)

    two_step = 2.0 * step
    N = len(np.atleast_1d(x_raw))
    J = np.zeros((N, 2, 2))
    J[:, 0, 0] = (xp - xm) / two_step
    J[:, 0, 1] = (xu - xd) / two_step
    J[:, 1, 0] = (yp - ym) / two_step
    J[:, 1, 1] = (yu - yd) / two_step
    return J


def _wcs_transform_batch(x_chip, y_chip, wcs_obj):
    """Apply per-chip WCS to 0-indexed chip coordinates."""
    xy = np.column_stack([np.asarray(x_chip, dtype=float),
                          np.asarray(y_chip, dtype=float)])
    sky = wcs_obj.all_pix2world(xy, 0)
    return sky[:, 0], sky[:, 1]


def _wcs_jacobian_batch(x_chip, y_chip, wcs_obj, step=0.1):
    """Compute d(RA,Dec)/d(x,y) Jacobian in deg/pix via central differences."""
    ra_xp, dec_xp = _wcs_transform_batch(x_chip + step, y_chip, wcs_obj)
    ra_xm, dec_xm = _wcs_transform_batch(x_chip - step, y_chip, wcs_obj)
    ra_yp, dec_yp = _wcs_transform_batch(x_chip, y_chip + step, wcs_obj)
    ra_ym, dec_ym = _wcs_transform_batch(x_chip, y_chip - step, wcs_obj)

    two_step = 2.0 * step
    N = len(np.atleast_1d(x_chip))
    J = np.zeros((N, 2, 2))
    J[:, 0, 0] = (ra_xp  - ra_xm)  / two_step
    J[:, 0, 1] = (ra_yp  - ra_ym)  / two_step
    J[:, 1, 0] = (dec_xp - dec_xm) / two_step
    J[:, 1, 1] = (dec_yp - dec_ym) / two_step
    return J


def _propagate_cov_2x2(cov_xy, J_batch):
    """Propagate 2×2 xy covariance through Jacobian J."""
    return J_batch @ cov_xy @ J_batch.transpose(0, 2, 1)


def xy2radec(x, y, r0, d0):
    """RA,Dec offsets to absolute RA,Dec."""
    xrad = np.radians(x)
    yrad = np.radians(y)
    r0rad = np.radians(r0)
    d0rad = np.radians(d0)
    cosd0 = np.cos(d0rad)
    sind0 = np.sin(d0rad)
    dr = np.arctan2(xrad, (cosd0 - yrad * sind0))
    ra = r0 + np.degrees(dr)
    tande = np.arctan2(np.cos(dr) * (sind0 + yrad * cosd0), (cosd0 - yrad * sind0))
    dec = np.degrees(tande)
    return ra, dec


def _apply_gdc_wcs(records, gdc, header=None, verbose=False):
    """Apply GDC and WCS corrections in-place to a list of StarRecords."""
    from astropy.wcs import WCS
    n = len(records)
    if n == 0:
        return

    if gdc is not None:
        x_raw = np.array([r.x for r in records])
        y_raw = np.array([r.y for r in records])
        x_gdc, y_gdc, mc = apply_gdc(x_raw, y_raw, gdc)
        J_gdc = _gdc_jacobian_batch(x_raw, y_raw, gdc)
        for i, r in enumerate(records):
            r._x_gdc = float(x_gdc[i])
            r._y_gdc = float(y_gdc[i])
            r._mc = float(mc[i])
            if r.cov is not None:
                cov_xy = r.cov[1:3, 1:3]
                r._cov_gdc = J_gdc[i] @ cov_xy @ J_gdc[i].T
            else:
                r._cov_gdc = None

    if header is not None:
        try:
            wcs_obj = WCS(header)
            x_raw = np.array([r.x for r in records])
            y_raw = np.array([r.y for r in records])
            ra, dec = _wcs_transform_batch(x_raw, y_raw, wcs_obj)
            J_wcs = _wcs_jacobian_batch(x_raw, y_raw, wcs_obj)
            for i, r in enumerate(records):
                r._ra = float(ra[i])
                r._dec = float(dec[i])
                if r.cov is not None:
                    cov_xy = r.cov[1:3, 1:3]
                    r._cov_radec = J_wcs[i] @ cov_xy @ J_wcs[i].T
                else:
                    r._cov_radec = None
        except Exception as e:
            if verbose:
                print(f"WCS application failed: {e}")


def write_gdc_auxiliary_files(image_path, records, gdc, output_dir=None):
    """
    Produce <image>_GDC_vals.csv and <image>_GDC_vals_REFCOORDS.csv.
    This matches the logic in GaiaJWSTmod.py.
    """
    stem = os.path.basename(image_path).replace(".fits", "")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))
    
    def _get_vals_and_derivs(xy_1based):
        # apply_gdc expects 0-based
        xy_0based = xy_1based - 1.0
        ux, uy, _ = apply_gdc(xy_0based[:,0], xy_0based[:,1], gdc)
        J = _gdc_jacobian_batch(xy_0based[:,0], xy_0based[:,1], gdc, step=0.001)
        return ux, uy, J

    # 1. GDC_vals.csv for detected stars
    if records:
        xy_1based = np.array([[r.x + 1.0, r.y + 1.0] for r in records])
        ux, uy, J = _get_vals_and_derivs(xy_1based)
        
        path_vals = os.path.join(output_dir, f"{stem}_GDC_vals.csv")
        # Match pandas to_csv() default format with index
        with open(path_vals, 'w') as f:
            f.write(",XC,YC,dXC_dXR,dXC_dYR,dYC_dXR,dYC_dYR\n")
            for i in range(len(ux)):
                f.write(f"{i},{ux[i]:.12f},{uy[i]:.12f},{J[i,0,0]:.12f},{J[i,0,1]:.12f},{J[i,1,0]:.12f},{J[i,1,1]:.12f}\n")

    # 2. GDC_vals_REFCOORDS.csv for reference points
    ref_coords = []
    with fits.open(image_path) as hdul:
        for hdu in hdul:
            if 'CRPIX1' in hdu.header:
                ref_coords.append([hdu.header['CRPIX1'], hdu.header['CRPIX2']])
    
    # Central pixel (match GaiaJWSTmod 1024.5, 1024.5)
    if [1024.5, 1024.5] not in ref_coords:
        ref_coords.append([1024.5, 1024.5])
    
    xy_ref = np.array(ref_coords)
    ux_ref, uy_ref, J_ref = _get_vals_and_derivs(xy_ref)
    
    path_ref = os.path.join(output_dir, f"{stem}_GDC_vals_REFCOORDS.csv")
    with open(path_ref, 'w') as f:
        f.write(",XC,YC,dXC_dXR,dXC_dYR,dYC_dXR,dYC_dYR,XR,YR\n")
        for i in range(len(ux_ref)):
            f.write(f"{i},{ux_ref[i]:.12f},{uy_ref[i]:.12f},{J_ref[i,0,0]:.12f},{J_ref[i,0,1]:.12f},{J_ref[i,1,0]:.12f},{J_ref[i,1,1]:.12f},{xy_ref[i,0]},{xy_ref[i,1]}\n")


def _run_nircam_meta(image_path, psf_cube, xs, ys, psf_scale, gdc_map, *,
                     n_passes, hmin, fmin=None, fmin_thresh=5.0, mag_limit=None,
                     hw, sky_inner, sky_outer,
                     gain, read_noise, zero_point, exposure_time,
                     verbose, return_residual,
                     input_catalog_path, inject, out_dir, conc_limit=0.9, **fit_kwargs):
    """Process a NIRCam 8-chip meta mosaic by iterating over each 2048×2048 sub-chip.

    Translation of the multi-chip measurement loop in Fortran jwst1pass
    (boundary-polygon chip assignment, lines 16868-16874, applied here by
    slicing the 8 rectangular sub-images defined in _NIRCAM_META_CHIP_BOUNDS).

    For each chip the sub-image is extracted from the full 8192×4096 mosaic,
    photometry is run independently, and the returned pixel coordinates are
    translated back to the full-mosaic frame before combining.

    Returns:
        all_records : list of StarRecord with full-mosaic x,y coordinates
        residuals   : dict mapping chip_number (1-8) → residual sub-array
        var_maps    : dict mapping chip_number (1-8) → variance sub-array
        header      : FITS header from the mosaic (for WCS application)
    """
    from .multipass import run_photometry

    full_data, full_mask, header = load_nircam_meta_full(image_path, verbose=verbose)

    noise_info = _get_noise_info(header, header, 'NIRCAM')
    _gain      = gain      if gain      is not None else noise_info['hardware_gain']
    _rn        = read_noise if read_noise is not None else noise_info['read_noise']

    input_stars = None
    if input_catalog_path is not None:
        input_stars = read_art_xym(input_catalog_path)

    all_records = []
    residuals = {}
    var_maps = {}

    # AB magnitude zero point from pixel solid angle (same for all 8 sub-chips)
    _pixar_sr = header.get('PIXAR_SR', None)
    if _pixar_sr is not None and _pixar_sr > 0:
        _zp_ab = -2.5 * np.log10(_pixar_sr * 1e6 / 3631.0)
    else:
        _zp_ab = np.nan
    if verbose:
        if np.isfinite(_zp_ab):
            print(f"  ZP_AB         : {_zp_ab:.4f}  (PIXAR_SR={_pixar_sr:.4e} sr)")
        else:
            print(f"  ZP_AB         : N/A  (PIXAR_SR not in header)")

    # Effective fmin: mag_limit interpreted in AB magnitudes via ZP_AB, floored at fmin_thresh
    _zp_for_mag = _zp_ab if np.isfinite(_zp_ab) else zero_point
    if fmin is not None:
        _fmin_effective = fmin
    elif mag_limit is not None:
        _fmin_from_mag = 10.0 ** ((_zp_for_mag - mag_limit) / 2.5)
        _fmin_effective = max(_fmin_from_mag, fmin_thresh)
    else:
        _fmin_effective = fmin_thresh

    if verbose:
        if fmin is not None:
            print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  (direct override)")
        elif mag_limit is not None:
            if _fmin_from_mag >= fmin_thresh:
                print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  (mag_limit={mag_limit:.2f} AB)")
            else:
                print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  "
                      f"(fmin_thresh floor; mag_limit={mag_limit:.2f} AB → {_fmin_from_mag:.4f} MJy/sr)")
        else:
            print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  (fmin_thresh)")

    for chip_idx, (col_lo, col_hi, row_lo, row_hi) in enumerate(_NIRCAM_META_CHIP_BOUNDS):
        chip_num = chip_idx + 1  # 1-based chip number (matches Fortran kru)

        sub_data = full_data[row_lo:row_hi, col_lo:col_hi].copy()
        sub_mask = full_mask[row_lo:row_hi, col_lo:col_hi].copy()

        # x_offset / y_offset tell run_photometry where to look up the spatially
        # varying PSF: the PSF grid is defined on the full detector frame.
        x_offset = float(col_lo)
        y_offset = float(row_lo)

        if verbose:
            print(f"\n  NIRCam meta chip {chip_num} "
                  f"(col {col_lo}:{col_hi}, row {row_lo}:{row_hi})")

        records, residual, var_map = run_photometry(
            sub_data, psf_cube, xs, ys, psf_scale,
            mask=sub_mask,
            n_passes=n_passes, hmin=hmin, fmin=_fmin_effective, hw=hw,
            sky_inner=sky_inner, sky_outer=sky_outer,
            gain=_gain, read_noise=_rn,
            zero_point=zero_point, exposure_time=exposure_time,
            x_offset=x_offset, y_offset=y_offset,
            verbose=verbose,
            input_stars=input_stars,
            inject=inject,
            conc_limit=conc_limit,
            **fit_kwargs
        )

        # Shift sub-image coordinates → full-mosaic coordinates
        for r in records:
            r.x += col_lo
            r.y += row_lo
            r._chip_ext  = chip_num
            r._instrument = 'NIRCAM'
            r._zp_ab = _zp_ab

        all_records.extend(records)
        var_maps[chip_num] = var_map  # always available for diagnostics
        if return_residual:
            residuals[chip_num] = residual

    return all_records, residuals, var_maps, header


def run_photometry_fits(image_path, psf_path=None, gdc_path=None, zero_point=0.0,
                        n_passes=2, hmin=5.0, fmin=None,
                        fmin_thresh=5.0, mag_limit=None, hw=5,
                        sky_inner=4, sky_outer=8, verbose=False,
                        return_residual=True, lib_dir=None,
                        input_catalog_path=None, out_dir=None,
                        inject=False, exposure_time=None,
                        conc_limit=0.9, **kwargs):
    """High-level driver to run photometry on all chips of a JWST image."""
    from .multipass import run_photometry
    
    # 1. Resolve Instrument and Library Paths
    with fits.open(image_path) as hdul:
        primary_hdr = hdul[0].header
        instrume = primary_hdr.get('INSTRUME', '').strip().upper()
        detector = primary_hdr.get('DETECTOR', '').strip().upper()
        if exposure_time is None:
            # MJy/sr is a rate unit — no exposure-time division in magnitude formula.
            if instrume in ('NIRISS', 'MIRI'):
                exposure_time = 1.0
            else:
                exposure_time = primary_hdr.get('EFFEXPTM', 1.0)
    
    det_prefix = _DETECTOR_PREFIX.get((instrume, detector), instrume)

    # 2. Resolve PSF
    _psf_path = psf_path
    if _psf_path is None and lib_dir is not None:
        # NIRCam files live under lib/STDPSFs/NIRCam/; other instruments under their prefix dir.
        if instrume == 'NIRCAM':
            _psf_path = os.path.join(lib_dir, 'STDPSFs', 'NIRCam')
        else:
            _psf_path = os.path.join(lib_dir, 'STDPSFs', det_prefix)

    if _psf_path is None:
        raise ValueError("Must provide psf_path or lib_dir")

    if os.path.isdir(_psf_path):
        psf_file = find_psf(_psf_path, primary_hdr)
    else:
        psf_file = _psf_path

    if psf_file is None or not os.path.exists(psf_file):
        raise FileNotFoundError(f"PSF file not found: {psf_file}")

    psf_cube, xs, ys, psf_scale, _ = load_psf_cube(psf_file)

    # 3. Resolve GDC
    gdc_map = None
    _gdc_path = gdc_path
    if _gdc_path is None and lib_dir is not None:
        if instrume == 'NIRCAM':
            _gdc_dir = os.path.join(lib_dir, 'STDGDCs', 'NIRCam')
        else:
            _gdc_dir = os.path.join(lib_dir, 'STDGDCs', det_prefix)
            if not os.path.isdir(_gdc_dir):
                _gdc_dir = os.path.join(lib_dir, 'STDGDCs')
        _gdc_path = find_gdc(_gdc_dir, primary_hdr) if os.path.isdir(_gdc_dir) else None
    elif _gdc_path is not None and os.path.isdir(_gdc_path):
        _gdc_path = find_gdc(_gdc_path, primary_hdr)

    filt = _extract_filter(primary_hdr, instrume)
    if verbose:
        print(f"Instrument: {instrume}/{detector}, Filter: {filt}")
        print(f"Auto-selected PSF: {os.path.basename(psf_file)}")
        if _gdc_path:
            print(f"GDC file: {os.path.basename(_gdc_path)}")

    if _gdc_path is None or not os.path.exists(_gdc_path):
        raise FileNotFoundError(f"GDC file not found or missing for {instrume}/{detector} {filt}")

    if verbose:
        print(f"Loading GDC: {_gdc_path}")
    gdc_map = load_stdgdc(_gdc_path)

    # 4a. NIRCam 8-chip meta mosaic — special multi-chip path
    # (Fortran jwst1pass lines 14618-15006; INSTu=29, Ks=8)
    if instrume == 'NIRCAM':
        with fits.open(image_path) as _hdul:
            _naxis1_sci = _hdul[1].header.get('NAXIS1', 0) if len(_hdul) > 1 else 0
        if _naxis1_sci == 8192:
            if verbose:
                print("NIRCam meta mosaic detected (8192×4096) — processing 8 chips")
            all_records, residuals, var_maps, _meta_header = _run_nircam_meta(
                image_path, psf_cube, xs, ys, psf_scale, gdc_map,
                n_passes=n_passes, hmin=hmin, fmin=fmin,
                fmin_thresh=fmin_thresh, mag_limit=mag_limit, hw=hw,
                sky_inner=sky_inner, sky_outer=sky_outer,
                gain=kwargs.get('gain'), read_noise=kwargs.get('read_noise'),
                zero_point=zero_point, exposure_time=exposure_time,
                verbose=verbose, return_residual=return_residual,
                input_catalog_path=input_catalog_path, inject=inject,
                out_dir=out_dir, conc_limit=conc_limit,
                **{k: v for k, v in kwargs.items()
                   if k not in ('gain', 'read_noise', 'dq_flags')}
            )
            _apply_gdc_wcs(all_records, gdc_map,
                           header=_meta_header, verbose=verbose)
            write_gdc_auxiliary_files(image_path, all_records, gdc_map,
                                      output_dir=out_dir)
            return all_records, residuals, var_maps, psf_file, _gdc_path

    # 4b. Chips and Noise (single-chip path: NIRISS, MIRI, NIRCam 2048×2048)
    chips = get_chip_config(instrume, detector)
    all_records = []
    residuals = {}
    var_maps = {}

    # 5. Run photometry per chip
    for sci_ext, dq_ext, y_offset in chips:
        with fits.open(image_path) as _hdul:
            _sci_hdr = _hdul[sci_ext].header
        _noise_info = _get_noise_info(primary_hdr, _sci_hdr, instrume)

        if verbose:
            print(f"\nChip: sci_ext={sci_ext}, dq_ext={dq_ext}, y_offset={y_offset}")
            _g_src = 'header' if _noise_info['gain_from_header'] else 'default'
            _rn_src = 'header' if _noise_info['rn_from_header'] else 'default'
            _g_note = (' [image in electrons — noise gain = 1.0]'
                       if _noise_info['data_in_electrons'] else ' [target DN]')
            print(f"  BUNIT         : {_noise_info['bunit']}")
            print(f"  Hardware gain : {_noise_info['hardware_gain']:.4f} e-/DN  [{_g_src}]")
            print(f"  Noise gain    : {_noise_info['effective_gain']:.4f}{_g_note}")
            print(f"  Read noise    : {_noise_info['read_noise']:.2f} e-  [{_rn_src}]")

        (data, gain_hdr, rn_hdr, mask, header, 
         x_off, y_off, pipeline_noise_map, gain_map, read_noise_map) = load_image(image_path, ext=sci_ext, dq_ext=dq_ext, 
                                                                                 dq_flags=kwargs.get('dq_flags'),
                                                                                 lib_dir=lib_dir, verbose=verbose)
        
        # Noise params (allow override)
        gain = kwargs.get('gain')
        if gain is None: gain = gain_hdr
        
        read_noise = kwargs.get('read_noise')
        if read_noise is None: read_noise = rn_hdr
        
        # Filter kwargs to avoid unexpected argument error
        fit_kwargs = kwargs.copy()
        fit_kwargs.pop('dq_flags', None)
        fit_kwargs.pop('gain', None)
        fit_kwargs.pop('read_noise', None)

        input_stars = None
        if input_catalog_path is not None:
            # Read X, Y, Mag from artificial star file
            input_stars = read_art_xym(input_catalog_path)

        # AB magnitude zero point from pixel solid angle
        _pixar_sr = _sci_hdr.get('PIXAR_SR', primary_hdr.get('PIXAR_SR', None))
        if _pixar_sr is not None and _pixar_sr > 0:
            _zp_ab = -2.5 * np.log10(_pixar_sr * 1e6 / 3631.0)
        else:
            _zp_ab = np.nan
        if verbose:
            if np.isfinite(_zp_ab):
                print(f"  ZP_AB         : {_zp_ab:.4f}  (PIXAR_SR={_pixar_sr:.4e} sr)")
            else:
                print(f"  ZP_AB         : N/A  (PIXAR_SR not in header)")

        # Effective fmin: mag_limit interpreted in AB magnitudes via ZP_AB, floored at fmin_thresh
        _zp_for_mag = _zp_ab if np.isfinite(_zp_ab) else zero_point
        if fmin is not None:
            _fmin_effective = fmin
        elif mag_limit is not None:
            _fmin_from_mag = 10.0 ** ((_zp_for_mag - mag_limit) / 2.5)
            _fmin_effective = max(_fmin_from_mag, fmin_thresh)
        else:
            _fmin_effective = fmin_thresh

        if verbose:
            if fmin is not None:
                print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  (direct override)")
            elif mag_limit is not None:
                if _fmin_from_mag >= fmin_thresh:
                    print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  (mag_limit={mag_limit:.2f} AB)")
                else:
                    print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  "
                          f"(fmin_thresh floor; mag_limit={mag_limit:.2f} AB → {_fmin_from_mag:.4f} MJy/sr)")
            else:
                print(f"  fmin          : {_fmin_effective:.4f} MJy/sr  (fmin_thresh)")

        # Data is in MJy/sr for all instruments — flux from fitter is already in MJy/sr.
        flux_unit_conv = 1.0
        records, residual, var_map = run_photometry(
            data, psf_cube, xs, ys, psf_scale, mask=mask,
            n_passes=n_passes, hmin=hmin, fmin=_fmin_effective, hw=hw,
            sky_inner=sky_inner, sky_outer=sky_outer,
            gain=gain, read_noise=read_noise, zero_point=zero_point,
            exposure_time=exposure_time,
            x_offset=x_off, y_offset=y_offset,
            verbose=verbose, pipeline_noise_map=pipeline_noise_map,
            gain_map=gain_map, read_noise_map=read_noise_map,
            input_stars=input_stars,
            inject=inject, flux_unit_conversion=flux_unit_conv,
            conc_limit=conc_limit,
            **fit_kwargs
        )
        
        # Track chip extension in record
        for r in records:
            r._chip_ext = sci_ext
            r._instrument = instrume
            r._filter = filt
            r._zp_ab = _zp_ab
            
        all_records.extend(records)
        var_maps[sci_ext] = var_map  # always available for diagnostics
        if return_residual:
            residuals[sci_ext] = residual
    
    # 6. Apply corrections to all records
    # Note: _apply_gdc_wcs needs updating if it should handle multi-header
    # For now we use the header of the last chip (usually same WCS base)
    _apply_gdc_wcs(all_records, gdc_map, header=header, verbose=verbose)

    # 7. Produce GDC auxiliary files (compatibility with GaiaJWSTmod)
    write_gdc_auxiliary_files(image_path, all_records, gdc_map, output_dir=out_dir)
    
    return all_records, residuals, var_maps, psf_file, _gdc_path


def catalog_to_table(records, zero_point=25.0):
    """Convert a list of StarRecord to an astropy Table with all columns."""
    from astropy.table import Table
    if not records:
        return Table(names=['x', 'y', 'flux', 'flux_err', 'sky', 'sky_err', 'mag', 'mag_err',
                            'mag_ab', 'q', 'Q', 'chi2', 'pass_number', 'n_iter', 'converged'],
                     dtype=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                            'f8', 'f8', 'f8', 'f8', 'i4', 'i4', 'b1'])
    
    def _c(r, i, j):
        return r.cov[i, j] if r.cov is not None else np.nan

    rows = []
    for r in records:
        _zp_ab = getattr(r, '_zp_ab', np.nan)
        _mag_ab = (-2.5 * np.log10(r.flux) + _zp_ab
                   if (r.flux > 0 and np.isfinite(_zp_ab)) else np.nan)
        row = {
            'x': r.x, 'y': r.y, 'flux': r.flux, 'flux_err': r.flux_err,
            'sky': r.sky, 'sky_err': r.sky_err, 'mag': r.mag, 'mag_err': r.mag_err,
            'mag_ab': _mag_ab,
            'q': r.qfit, 'Q': r.Qfit, 'chi2': r.chi2, 'pass_number': r.pass_number,
            'n_iter': r.n_iter, 'converged': r.converged,
            'psf_frac': r.psf_frac, 'psf_peak': r.psf_peak, 'peak': r.peak,
            'central_res': r.central_res, 'n_sat': r.n_sat,
            'n_neighbors': r.n_neighbors,
            'dist_nearest': r.dist_nearest, 'dist_nearest_brighter': r.dist_nearest_brighter,
            'chi2_scale': r.chi2_scale, 'delta_max': r.delta_max,
            'cov_ff': _c(r, 0, 0), 'cov_xx': _c(r, 1, 1), 'cov_yy': _c(r, 2, 2), 'cov_ss': _c(r, 3, 3),
            'cov_fx': _c(r, 0, 1), 'cov_fy': _c(r, 0, 2), 'cov_fs': _c(r, 0, 3),
            'cov_xy': _c(r, 1, 2), 'cov_xs': _c(r, 1, 3), 'cov_ys': _c(r, 2, 3)
        }
        
        # GDC columns
        x_gdc = getattr(r, '_x_gdc', np.nan)
        y_gdc = getattr(r, '_y_gdc', np.nan)
        mc = getattr(r, '_mc', 0.0)
        cov_gdc = getattr(r, '_cov_gdc', None)
        
        row.update({
            'x_gdc': x_gdc, 'y_gdc': y_gdc, 
            'mag_gdc': r.mag + mc,
            'cov_xx_gdc': cov_gdc[0, 0] if cov_gdc is not None else np.nan,
            'cov_yy_gdc': cov_gdc[1, 1] if cov_gdc is not None else np.nan,
            'cov_xy_gdc': cov_gdc[0, 1] if cov_gdc is not None else np.nan,
        })

        # WCS columns
        ra = getattr(r, '_ra', np.nan)
        dec = getattr(r, '_dec', np.nan)
        cov_radec = getattr(r, '_cov_radec', None)
        
        row.update({
            'ra': ra, 'dec': dec,
            'cov_ra_ra': cov_radec[0, 0] if cov_radec is not None else np.nan,
            'cov_dec_dec': cov_radec[1, 1] if cov_radec is not None else np.nan,
            'cov_ra_dec': cov_radec[0, 1] if cov_radec is not None else np.nan,
        })
        
        rows.append(row)
        
    tab = Table(rows)
    tab.meta['ZP'] = zero_point
    tab.meta['ZP_AB'] = float(getattr(records[0], '_zp_ab', np.nan)) if records else np.nan
    
    # Add descriptions for q and Q
    if 'q' in tab.colnames:
        tab['q'].description = 'Quality of fit (5x5 core window, matching Fortran)'
    if 'Q' in tab.colnames:
        tab['Q'].description = 'Quality of fit (Full fit window)'
        
    return tab


def write_fortran_catalog(records, path, fortran_metadata=None):
    """
    Write records in a 15-column space-separated format matching jwst1pass 
    Fortran output for 'uvmqxyrdabgcenW'.
    
    Includes an optional fortran_metadata dictionary to replicate the 
    extensive logging block of the Fortran code.
    """
    import os
    # Try to extract instrument/filter for the comment line
    instrume = "UNKNOWN"
    filt = "UNKNOWN"
    if records:
        instrume = getattr(records[0], '_instrument', "NIRISS")
        filt = getattr(records[0], '_filter', "F200W")

    with open(path, 'w') as f:
        # 1. Legacy Argument/Metadata block
        if fortran_metadata:
            f.write("#\n")
            f.write("#--------------------------------------------\n")
            f.write("# ARGUMENTS\n")
            f.write("#--------------------------------------------\n")
            args_list = fortran_metadata.get('args', [])
            for i, arg in enumerate(args_list):
                f.write(f"#  ARG{i:04d}: {arg}\n")
            f.write("#\n")
            f.write("#================================\n")
            f.write("# BASIC STAR-FINDING PARAMETERS\n")
            f.write("#\n")
            f.write(f"# FILENAME: {fortran_metadata.get('filename', 'UNKNOWN')}\n")
            f.write(f"#   NAXIS1:  {fortran_metadata.get('naxis1', 0)}\n")
            f.write(f"#   NAXIS2:  {fortran_metadata.get('naxis2', 0)}\n")
            f.write("#\n")

        # 2. Compatibility headers
        f.write(f"# : GDC=lib/STDGDCs/{instrume}/STDGDC_{instrume}_{filt}.fits\n")
        f.write("# rDAT: 0.0\n") 
        # The parser (make_XYmqxyrd) calculates cut_ind from the first 8 columns of this header.
        # Lengths: 13, 12, 9, 9, 10, 10, 15, 15. Sum=93. 7 spaces. cut_ind = 100.
        header = "#..u......... ..v......... ..Mag.... ..Qfit... ..X....... ..Y....... ..RA........... ..Dec..........      DUM1   DUM2   DUM3   DUM4   DUM5   DUM6   DUM7\n"
        f.write(header)
        
        # 3. Data Rows
        zp = fortran_metadata.get('zero_point', 0.0) if fortran_metadata else 0.0
        for r in records:
            x_gdc = getattr(r, '_x_gdc', 0.0)
            y_gdc = getattr(r, '_y_gdc', 0.0)
            ra = getattr(r, '_ra', 0.0)
            dec = getattr(r, '_dec', 0.0)
            
            # instrumental mag should be negative for xym2pm compatibility
            if np.isnan(r.mag):
                inst_mag = 99.99
            else:
                inst_mag = r.mag - zp
            
            # Extract uncertainties from covariance matrix (flux, x, y, sky)
            # Index 1 is X, Index 2 is Y
            sx = 0.0; sy = 0.0; rho = 0.0
            if r.cov is not None and np.all(np.isfinite(r.cov)):
                sx = np.sqrt(max(r.cov[1, 1], 0.0))
                sy = np.sqrt(max(r.cov[2, 2], 0.0))
                if sx > 0 and sy > 0:
                    rho = r.cov[1, 2] / (sx * sy)
                    rho = np.clip(rho, -0.999, 0.999)

            # Format MUST match the header widths for make_XYmqxyrd to slice correctly.
            # JA standard columns 9, 10, 11 are sx, sy, rho.
            line = (f" {x_gdc:12.4f} {y_gdc:11.4f} {inst_mag:8.4f} {r.qfit:8.4f} "
                    f"{r.x + 1.0:9.4f} {r.y + 1.0:9.4f} {ra:14.8f} {dec:14.8f} "
                    f" {sx:10.6f} {sy:10.6f} {rho:10.6f}    0.0    0.0    0.0    0.0\n")
            f.write(line)


def write_XYmqxyrd_catalog(records, path, fortran_metadata=None):
    """
    Write records in an 8-column space-separated format matching jwst1pass 
    Fortran output for '.XYmqxyrd'.
    """
    # Try to extract instrument/filter for the comment line
    instrume = "UNKNOWN"
    filt = "UNKNOWN"
    if records:
        instrume = getattr(records[0], '_instrument', "NIRISS")
        filt = getattr(records[0], '_filter', "F200W")

    with open(path, 'w') as f:
        # 1. Legacy Argument/Metadata block
        if fortran_metadata:
            f.write("#\n")
            f.write("#--------------------------------------------\n")
            f.write("# ARGUMENTS\n")
            f.write("#--------------------------------------------\n")
            args_list = fortran_metadata.get('args', [])
            for i, arg in enumerate(args_list):
                f.write(f"#  ARG{i:04d}: {arg}\n")
            f.write("#\n")
            f.write("#================================\n")
            f.write("# BASIC STAR-FINDING PARAMETERS\n")
            f.write("#\n")
            f.write(f"# FILENAME: {fortran_metadata.get('filename', 'UNKNOWN')}\n")
            f.write(f"#   NAXIS1:  {fortran_metadata.get('naxis1', 0)}\n")
            f.write(f"#   NAXIS2:  {fortran_metadata.get('naxis2', 0)}\n")
            f.write("#\n")

        # 2. Compatibility headers
        f.write(f"# : GDC=lib/STDGDCs/{instrume}/STDGDC_{instrume}_{filt}.fits\n")
        f.write("# rDAT: 0.0\n") 
        header = "#..u......... ..v......... ..Mag.... ..Qfit... ..X....... ..Y....... ..RA........... ..Dec..........  \n"
        f.write(header)
        
        # 3. Data Rows
        zp = fortran_metadata.get('zero_point', 0.0) if fortran_metadata else 0.0
        for r in records:
            x_gdc = getattr(r, '_x_gdc', 0.0)
            y_gdc = getattr(r, '_y_gdc', 0.0)
            ra = getattr(r, '_ra', 0.0)
            dec = getattr(r, '_dec', 0.0)
            
            if np.isnan(r.mag):
                inst_mag = 99.99
            else:
                inst_mag = r.mag - zp

            # Extract uncertainties from covariance matrix (flux, x, y, sky)
            sx = 0.0; sy = 0.0; rho = 0.0
            if r.cov is not None and np.all(np.isfinite(r.cov)):
                sx = np.sqrt(max(r.cov[1, 1], 0.0))
                sy = np.sqrt(max(r.cov[2, 2], 0.0))
                if sx > 0 and sy > 0:
                    rho = r.cov[1, 2] / (sx * sy)
                    rho = np.clip(rho, -0.999, 0.999)
            
            line = (f" {x_gdc:12.4f} {y_gdc:11.4f} {inst_mag:8.4f} {r.qfit:8.4f} "
                    f"{r.x + 1.0:9.4f} {r.y + 1.0:9.4f} {ra:14.8f} {dec:14.8f} "
                    f" {sx:10.6f} {sy:10.6f} {rho:10.6f}\n")
            f.write(line)


def read_art_xym(path):
    """
    Read an artificial star file (_ART.xym) containing X, Y, Mag.
    Converts 1-based input coordinates to 0-based internal coordinates.
    """
    # Skip comments
    data = np.genfromtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    # Coordinates in .xym files are 1-based; subtract 1.0 for internal 0-based use.
    if data.shape[0] > 0:
        data[:, 0] -= 1.0  # X
        data[:, 1] -= 1.0  # Y
        
    # Returns a list of (x, y, mag)
    return data

