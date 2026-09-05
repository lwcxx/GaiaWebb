"""FITS image loader and PSF loader for JWST (MIRI/NIRISS)."""

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# Instrument/detector → prefix used in STDPSF filenames
_DETECTOR_PREFIX = {
    ('MIRI', 'MIRIM'): 'MIRI',
    ('NIRISS', 'NIS'): 'NIRISS',
}

def _extract_filter(header, instrume):
    """Extract the science filter name from a JWST header."""
    instrume = instrume.strip().upper()
    filt = header.get('FILTER', '').strip().upper()
    pupil = header.get('PUPIL', '').strip().upper()
    if instrume == 'NIRISS' and filt == 'CLEAR':
        return pupil
    return filt


def get_chip_config(instrume, detector):
    """Return per-chip configuration (stub for JWST)."""
    return [(1, 2, 0.0)] # sci_ext, dq_ext, y_offset


def load_image(image_path, ext=1, dq_ext=None, dq_flags=None, lib_dir=None):
    """Load a JWST _cal image, apply unit conversion, and return noise/mask info."""

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
        
        # Unit conversion: Disable conversion to DN to match Fortran instrumental scale (MJy/sr)
        # photmjsr = header.get('PHOTMJSR', 1.0)
        # effexptm = header.get('EFFEXPTM', 1.0)
        # xfactor = effexptm / photmjsr
        # units = header.get('BUNIT', '').strip().upper()
        # if units == 'MJY/SR':
        #    data *= xfactor

        # Load the ERR extension to provide a pipeline-based noise map
        pipeline_noise_map = None
        err_ext = None
        for i in range(1, len(hdul)):
            if 'ERR' in hdul[i].name.upper():
                err_ext = i
                break
        if err_ext is not None:
            err_data = hdul[err_ext].data.astype(np.float64)
            # No scaling of ERR data either
            # if units == 'MJY/SR':
            #    err_data *= xfactor
            # Variance = ERR^2
            pipeline_noise_map = (err_data**2)

            # Mask invalid/negative variances
            pipeline_noise_map[~np.isfinite(pipeline_noise_map)] = 1e9
            pipeline_noise_map[pipeline_noise_map <= 0] = 1e9

        x_offset, y_offset = _detector_offsets(instrume, detector, ext)

        return (data, noise_info['effective_gain'], noise_info['read_noise'],
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
        # MIRI in Fortran uses the ERR extension directly (sig2 = pixe**2)
        # after XFACTOR conversion. This implies an effective gain of 1.0.
        gain_hdr = 1.0    
        rn_hdr = 5.0      # Fallback RN if ERR is missing
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


def _get_gain_map(instrume, shape, header=None, lib_dir=None):
    """Return a 2D spatial gain map for the instrument, prioritizing reference files."""
    ny, nx = shape
    
    # 1. Try to load from reference file if provided
    if header is not None and lib_dir is not None:
        r_gain = header.get('R_GAIN', '')
        if r_gain and r_gain.startswith('crds://'):
            ref_name = r_gain.replace('crds://', '')
            ref_path = os.path.join(lib_dir, 'R_Gain', ref_name)
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
            ref_path = os.path.join(lib_dir, 'R_READNO', ref_name)
            if os.path.exists(ref_path):
                try:
                    with fits.open(ref_path) as hdul:
                        rn_map = hdul[1].data.astype(np.float64)
                        if rn_map.shape == (ny, nx):
                            # Units check: NIRISS reference RN is in DN.
                            # We need electrons: RN(e-) = RN(DN) * Gain(e-/DN)
                            if instrume == 'NIRISS' and hdul[1].header.get('BUNIT') == 'DN':
                                if gain_map is not None:
                                    return rn_map * gain_map
                                else:
                                    return rn_map * 1.9835 # Fallback mean gain
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
    
    # Try multiple common layouts
    candidates = [
        os.path.join(psf_dir, 'STDPSFs', prefix, filename), # lib/STDPSFs/NIRISS/...
        os.path.join(psf_dir, 'STDPSFs', filename),         # lib/STDPSFs/...
        os.path.join(psf_dir, prefix, filename),           # lib/NIRISS/...
        os.path.join(psf_dir, filename)                    # lib/...
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
        
    raise FileNotFoundError(f"No PSF found for {instrume}/{detector} {filt} in {psf_dir}. \n"
                           f"Searched paths: {candidates}")


def find_gdc(gdc_dir, header):
    """Find the best-matching STDGDC file for a JWST header."""
    instrume = header.get('INSTRUME', '').strip().upper()
    detector = header.get('DETECTOR', '').strip().upper()
    filt = _extract_filter(header, instrume)
        
    prefix = _DETECTOR_PREFIX.get((instrume, detector), instrume)
    
    filename = f'STDGDC_{prefix}_{filt}.fits'
    path = os.path.join(gdc_dir, prefix, filename)
    
    if os.path.exists(path):
        return path
        
    path = os.path.join(gdc_dir, filename)
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


def _gdc_jacobian_batch(x_raw, y_raw, gdc, step=0.1):
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


def run_photometry_fits(image_path, psf_path=None, gdc_path=None, zero_point=25.0, 
                        n_passes=2, hmin=5.0, fmin=1000.0, hw=5, 
                        sky_inner=4, sky_outer=8, verbose=False, 
                        return_residual=True, lib_dir=None, 
                        input_catalog_path=None, **kwargs):
    """High-level driver to run photometry on all chips of a JWST image."""
    from .multipass import run_photometry
    
    # 1. Resolve Instrument and Library Paths
    with fits.open(image_path) as hdul:
        primary_hdr = hdul[0].header
        instrume = primary_hdr.get('INSTRUME', '').strip().upper()
        detector = primary_hdr.get('DETECTOR', '').strip().upper()
    
    det_prefix = _DETECTOR_PREFIX.get((instrume, detector), instrume)

    # 2. Resolve PSF
    _psf_path = psf_path
    if _psf_path is None and lib_dir is not None:
        _psf_path = os.path.join(lib_dir, 'STDPSFs', det_prefix)
    
    if _psf_path is None:
        raise ValueError("Must provide psf_path or lib_dir")

    if os.path.isdir(_psf_path):
        psf_file = find_psf(_psf_path, primary_hdr)
    else:
        psf_file = _psf_path
    
    psf_cube, xs, ys, psf_scale, _ = load_psf_cube(psf_file)
    
    # 3. Resolve GDC
    gdc = None
    _gdc_path = gdc_path
    if _gdc_path is None and lib_dir is not None:
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

    if _gdc_path and os.path.exists(_gdc_path):
        if verbose:
            print(f"Loading GDC: {_gdc_path}")
        gdc = load_stdgdc(_gdc_path)
            
    # 4. Chips and Noise
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
                                                                                 lib_dir=lib_dir)
        
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

        records, residual, var_map = run_photometry(
            data, psf_cube, xs, ys, psf_scale, mask=mask,
            n_passes=n_passes, hmin=hmin, fmin=fmin, hw=hw,
            sky_inner=sky_inner, sky_outer=sky_outer,
            gain=gain, read_noise=read_noise, zero_point=zero_point,
            x_offset=x_off, y_offset=y_off,
            verbose=verbose, pipeline_noise_map=pipeline_noise_map,
            gain_map=gain_map, read_noise_map=read_noise_map,
            input_stars=input_stars,
            **fit_kwargs
        )
        
        # Track chip extension in record
        for r in records:
            r._chip_ext = sci_ext
            r._instrument = instrume
            r._filter = filt
            
        all_records.extend(records)
        if return_residual:
            residuals[sci_ext] = residual
            var_maps[sci_ext] = var_map
    
    # 6. Apply corrections to all records
    # Note: _apply_gdc_wcs needs updating if it should handle multi-header
    # For now we use the header of the last chip (usually same WCS base)
    _apply_gdc_wcs(all_records, gdc, header=header, verbose=verbose)
    
    return all_records, residuals, var_maps, psf_file, _gdc_path


def catalog_to_table(records, zero_point=25.0):
    """Convert a list of StarRecord to an astropy Table with all columns."""
    from astropy.table import Table
    if not records:
        return Table(names=['x', 'y', 'flux', 'flux_err', 'sky', 'sky_err', 'mag', 'mag_err',
                            'q', 'Q', 'chi2', 'pass_number', 'n_iter', 'converged'],
                     dtype=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 
                            'f8', 'f8', 'f8', 'i4', 'i4', 'b1'])
    
    def _c(r, i, j):
        return r.cov[i, j] if r.cov is not None else np.nan

    rows = []
    for r in records:
        row = {
            'x': r.x, 'y': r.y, 'flux': r.flux, 'flux_err': r.flux_err,
            'sky': r.sky, 'sky_err': r.sky_err, 'mag': r.mag, 'mag_err': r.mag_err,
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
            # handle NaN by using a dummy large value (99.99 instrumental)
            if np.isnan(r.mag):
                inst_mag = 99.99
            else:
                inst_mag = r.mag - zp
            
            # Format MUST match the header widths for make_XYmqxyrd to slice correctly.
            line = (f" {x_gdc:12.4f} {y_gdc:11.4f} {inst_mag:8.4f} {r.qfit:8.4f} "
                    f"{r.x + 1.0:9.4f} {r.y + 1.0:9.4f} {ra:14.8f} {dec:14.8f} "
                    f"          0.0    0.0    0.0    0.0    0.0    0.0    0.0\n")
            f.write(line)


def read_art_xym(path):
    """Read an artificial star file (_ART.xym) containing X, Y, Mag."""
    # Skip comments
    data = np.genfromtxt(path, comments='#')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    # Returns a list of (x, y, mag)
    return data

