#!/usr/bin/env python3
"""
High-level driver script for jwst1pass (Python version).
"""

import os
import argparse
import numpy as np
from astropy.io import fits
from jwst1pass.io import run_photometry_fits, catalog_to_table, load_image, load_psf_cube, get_chip_config
from jwst1pass.diagnostics import (
    plot_diagnostics, plot_catalog_stats, plot_psf_residual_map
)
from jwst1pass.diagnostics_Q import (
    plot_diagnostics as plot_diagnostics_Q,
    plot_catalog_stats as plot_catalog_stats_Q,
    plot_psf_residual_map as plot_psf_residual_map_Q
)

def main():
    parser = argparse.ArgumentParser(
        description="Run jwst1pass on a JWST image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to input _cal.fits image")
    parser.add_argument("--psf", help="Path to STDPSF file (optional if lib_dir is provided)")
    parser.add_argument("--gdc", help="Path to STDGDC file (optional if lib_dir is provided)")
    parser.add_argument("--lib_dir", help="Path to library directory (lib/STDPSFs, lib/STDGDCs)")
    parser.add_argument("--out_dir", default=None, 
                        help="Directory for all output files. Defaults to the directory containing --image.")
    
    parser.add_argument("--fmin", type=float, default=None,
                        help="Direct flux threshold in MJy/sr (overrides --mag_limit/--fmin_thresh)")
    parser.add_argument("--fmin_thresh", type=float, default=5.0,
                        help="Hard lower bound on fmin in MJy/sr (default 5.0)")
    parser.add_argument("--mag_limit", type=float, default=None,
                        help="Faint instrumental magnitude limit; converted to MJy/sr using --zero_point")
    parser.add_argument("--hmin", type=float, default=5.0, help="Minimum S/N for discovery")
    parser.add_argument("--n_passes", type=int, default=2, help="Number of multi-pass iterations")
    parser.add_argument("--n_discovery_passes", type=int, default=1,
                        help="Passes that include new-source detection (default: n_passes-1)")
    parser.add_argument("--half_width", type=int, default=5, help="Fit window half-width")
    parser.add_argument("--sky_inner", type=int, default=4, help="Sky annulus inner radius")
    parser.add_argument("--sky_outer", type=int, default=8, help="Sky annulus outer radius")
    parser.add_argument("--gain", type=float, default=None, help="Detector gain e-/DN (auto-read if omitted)")
    parser.add_argument("--read_noise", type=float, default=None, help="Read noise e- (auto-read if omitted)")
    parser.add_argument("--zero_point", "--zp", type=float, default=0.0, help="Magnitude zero point (match Fortran)")
    
    parser.add_argument("--max_iter", type=int, default=1000, help="Max Newton iterations per star")
    parser.add_argument("--tol", type=float, default=1e-3, help="Position convergence tolerance")
    parser.add_argument("--sat_threshold", type=float, default=None, help="Saturation threshold (DN)")
    parser.add_argument("--sigma_clip_sigma", type=float, default=4.0, help="Sigma clipping threshold")
    parser.add_argument("--no_sigma_clip", action="store_true", help="Disable sigma clipping")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel worker threads (-1=all)")
    parser.add_argument("--exposure_time", type=float, default=None, help="Exposure time for mag normalization")

    parser.add_argument("--output", default=None, help="Output catalog file")
    parser.add_argument("--output_format", default=None, choices=['fits', 'ecsv', 'csv'], help="Output format")
    parser.add_argument("--residual", default=None, help="Residual image FITS path")
    parser.add_argument("--no_residual", action="store_true", help="Disable residual image output")
    
    parser.add_argument("--plot", action="store_true", help="Generate all diagnostic plots")
    parser.add_argument("--diagnostics", help="Path to diagnostics plot")
    parser.add_argument("--no_diagnostics", action="store_true", help="Disable diagnostic plot")
    parser.add_argument("--catalog_stats", help="Path to catalog stats plot")
    parser.add_argument("--no_catalog_stats", action="store_true", help="Disable catalog stats plot")
    parser.add_argument("--psf_residual_map", help="Path to PSF residual map")
    parser.add_argument("--no_psf_residual_map", action="store_true", help="Disable PSF residual map")
    
    parser.add_argument("--dq_flags", type=int, nargs="+", help="DQ flags to mask")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    
    # Compatibility with GaiaWebb (Fortran style)
    parser.add_argument("--input_catalog", help="Path to input star list (X, Y, Mag) to fit (e.g. _ART.xym)")
    parser.add_argument("--inject", action="store_true", help="Inject input_catalog stars into image (for ART tests)")
    parser.add_argument("--no_inject", action="store_true", help="Disable injection even if input_catalog is provided")
    parser.add_argument("--fortran_output", help="Path to save text catalog in Fortran-compatible format")
    
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.dirname(os.path.abspath(args.image))
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
        
    # Standard output names
    stem = os.path.basename(args.image).replace(".fits", "")
    
    def _resolve(path, default_name):
        if path is None:
            return os.path.join(args.out_dir, default_name)
        if os.path.isabs(path):
            return path
        # If path contains directory separators, treat as relative to CWD
        if os.sep in path or (os.altsep and os.altsep in path):
            return os.path.abspath(path)
        return os.path.join(args.out_dir, path)

    args.output = _resolve(args.output, f"{stem}_catalog.fits")
    args.fortran_output = _resolve(args.fortran_output, f"{stem}.uvmqxyrdabgcenW")
    
    if args.no_residual:
        args.residual = None
    else:
        args.residual = _resolve(args.residual, f"{stem}_residual.fits")
    
    # Initialize all diagnostic paths to None
    args.diagnostics = None
    args.diagnostics_Q = None
    args.catalog_stats = None
    args.catalog_stats_Q = None
    args.psf_residual_map = None
    args.psf_residual_map_Q = None

    if args.plot or args.diagnostics:
        if not args.no_diagnostics:
            args.diagnostics = _resolve(args.diagnostics, f"{stem}_diagnostics.png")
            args.diagnostics_Q = _resolve(None, f"{stem}_diagnostics_Q.png")

    if args.plot or args.catalog_stats:
        if not args.no_catalog_stats:
            args.catalog_stats = _resolve(args.catalog_stats, f"{stem}_catalog_stats.png")
            args.catalog_stats_Q = _resolve(None, f"{stem}_catalog_stats_Q.png")

    if args.plot or args.psf_residual_map:
        if not args.no_psf_residual_map:
            args.psf_residual_map = _resolve(args.psf_residual_map, f"{stem}_psf_residual_map.png")
            args.psf_residual_map_Q = _resolve(None, f"{stem}_psf_residual_map_Q.png")

    if args.verbose:
        print(f"Image   : {args.image}")
        print(f"out_dir : {args.out_dir}")
        print(f"lib_dir : {args.lib_dir}")

    # 1. Run Photometry
    # ART injection logic: default to True if input_catalog is provided
    inject_stars = False
    if args.input_catalog and not args.no_inject:
        inject_stars = True
    if args.inject: # Explicit override
        inject_stars = True

    try:
        stars, residuals, var_maps, psf_file, gdc_file = run_photometry_fits(
            args.image, psf_path=args.psf, gdc_path=args.gdc, lib_dir=args.lib_dir,
            fmin=args.fmin, fmin_thresh=args.fmin_thresh, mag_limit=args.mag_limit,
            hmin=args.hmin, n_passes=args.n_passes,
            n_discovery_passes=args.n_discovery_passes,
            hw=args.half_width, sky_inner=args.sky_inner, sky_outer=args.sky_outer,
            gain=args.gain, read_noise=args.read_noise,
            max_iter=args.max_iter, tol=args.tol,
            sat_threshold=args.sat_threshold if args.sat_threshold is not None else float('inf'),
            sigma_clip=not args.no_sigma_clip, sigma_clip_sigma=args.sigma_clip_sigma,
            n_jobs=args.n_jobs,
            zero_point=args.zero_point,
            verbose=args.verbose, dq_flags=args.dq_flags,
            return_residual=args.residual is not None,
            input_catalog_path=args.input_catalog,
            inject=inject_stars,
            exposure_time=args.exposure_time,
            out_dir=args.out_dir
        )
    except FileNotFoundError as e:
        print(f"SKIPPING {args.image}: {e}")
        return
    
    # 2. Convert to Table and save
    table = catalog_to_table(stars, zero_point=args.zero_point)
    
    fmt = args.output_format
    if fmt is None:
        ext = os.path.splitext(args.output)[1].lower()
        fmt = 'fits' if ext == '.fits' else ('ecsv' if ext == '.ecsv' else ('csv' if ext == '.csv' else 'fits'))

    if fmt == 'fits':
        table.write(args.output, overwrite=True)
    elif fmt == 'ecsv':
        table.write(args.output, format='ascii.ecsv', overwrite=True)
    elif fmt == 'csv':
        table.write(args.output, format='ascii.csv', overwrite=True)
        
    print(f"Saving catalog to: {args.output}")

    if args.fortran_output:
        import sys
        from jwst1pass.io import write_fortran_catalog, write_XYmqxyrd_catalog
        
        # Prepare legacy metadata
        with fits.open(args.image) as hdul:
            naxis1 = hdul[1].header.get('NAXIS1', 0)
            naxis2 = hdul[1].header.get('NAXIS2', 0)
            
        fortran_metadata = {
            'args': sys.argv,
            'filename': os.path.abspath(args.image),
            'naxis1': naxis1,
            'naxis2': naxis2,
            'zero_point': args.zero_point
        }
        
        write_fortran_catalog(stars, args.fortran_output, fortran_metadata=fortran_metadata)
        print(f"Saving Fortran-compatible catalog to: {args.fortran_output}")

        # Also save .XYmqxyrd by default
        xym_output = args.fortran_output.replace(".uvmqxyrdabgcenW", ".XYmqxyrd")
        write_XYmqxyrd_catalog(stars, xym_output, fortran_metadata=fortran_metadata)
        print(f"Saving XYmqxyrd catalog to: {xym_output}")
    
    # 3. Save Residual Image
    if args.residual is not None and residuals:
        with fits.open(args.image) as hdul:
            # Create a new HDU list for residual
            res_hdul = fits.HDUList([fits.PrimaryHDU()])
            for ext in sorted(residuals.keys()):
                res_data = residuals[ext]
                var_data = var_maps[ext]
                
                # Sci extension
                h = hdul[ext].header.copy()
                res_hdul.append(fits.ImageHDU(res_data.astype('float32'), header=h, name=f"RES_{ext}"))
                
                # Var extension
                h_var = h.copy()
                h_var['BUNIT'] = 'DN^2'
                res_hdul.append(fits.ImageHDU(var_data.astype('float32'), header=h_var, name=f"VAR_{ext}"))
                
            res_hdul.writeto(args.residual, overwrite=True)
            print(f"Residual image written to {args.residual} (SCI + VAR extensions)")

    # 4. Summary
    if stars:
        mags = table['mag']
        chip_ids = sorted({getattr(r, '_chip_ext', 1) for r in stars})
        print(f"\n====================================================")
        print(f"  Catalog summary  ({len(stars)} stars)")
        print(f"====================================================")
        print(f"  Magnitude range : {np.nanmin(mags):.2f} → {np.nanmax(mags):.2f}")
        for p in range(1, args.n_passes + 1):
            n_p = (table['pass_number'] == p).sum()
            print(f"  Pass breakdown  : pass {p}: {n_p}")

        if len(chip_ids) > 1:
            for cid in chip_ids:
                n_c = sum(1 for r in stars if getattr(r, '_chip_ext', 1) == cid)
                print(f"  Chip {cid:2d}         : {n_c} stars")

        print(f"  q (5x5) : median={np.nanmedian(table['q']):.3f}  "
              f"90th={np.nanpercentile(table['q'], 90):.3f}   (>0.5 = poor)")
        print(f"  Q (full): median={np.nanmedian(table['Q']):.3f}  "
              f"90th={np.nanpercentile(table['Q'], 90):.3f}")
        print(f"  chi²    : median={np.nanmedian(table['chi2']):.3f}  "
              f"90th={np.nanpercentile(table['chi2'], 90):.3f}   (~1.0 = ideal)")

        sn = table['flux'] / table['flux_err']
        print(f"  S/N   : median={np.nanmedian(sn):.1f}")

        n_sat = (table['n_sat'] > 0).sum()
        print(f"  Saturated (n_sat > 0)        : {n_sat} ({100*n_sat/len(stars):.1f}%)")

        n_crowd = (table['dist_nearest'] < 5).sum()
        print(f"  Crowded (brighter nbr <5px) : {n_crowd} ({100*n_crowd/len(stars):.1f}%)")

        n_fail = (table['converged'] == False).sum()
        print(f"  Did not converge (max_iter)   : {n_fail} ({100*n_fail/len(stars):.1f}%)")
        print(f"====================================================")

    # 5. Plots
    if stars:
        psf_cube, xs, ys, psf_scale, _ = load_psf_cube(psf_file)

        with fits.open(args.image) as hdul:
            primary_hdr = hdul[0].header
        instrume = primary_hdr.get('INSTRUME', '').strip().upper()
        detector = primary_hdr.get('DETECTOR', '').strip().upper()
        chips = get_chip_config(instrume, detector)

        # For NIRCam meta mosaic the residuals are keyed by chip number 1-8;
        # for single-chip instruments they are keyed by FITS extension (1).
        # Use the first available key for the primary diagnostic chip.
        _diag_chip_key = list(residuals.keys())[0] if residuals else 1

        # Load the image data for the first chip for diagnostics.
        # NIRCam meta: load the full mosaic and take chip 1 sub-image.
        def _load_diag_data(chip_key):
            if instrume == 'NIRCAM':
                with fits.open(args.image) as _hdul:
                    _naxis1 = _hdul[1].header.get('NAXIS1', 0) if len(_hdul) > 1 else 0
                if _naxis1 == 8192:
                    from jwst1pass.io import (load_nircam_meta_full,
                                              _NIRCAM_META_CHIP_BOUNDS)
                    full_data, full_mask, _ = load_nircam_meta_full(args.image)
                    col_lo, col_hi, row_lo, row_hi = _NIRCAM_META_CHIP_BOUNDS[chip_key - 1]
                    sub_data = full_data[row_lo:row_hi, col_lo:col_hi]
                    sub_mask = full_mask[row_lo:row_hi, col_lo:col_hi]
                    return sub_data, 2.05, 15.0, sub_mask, 0.0, 0.0
            # Default single-chip path
            sci_ext = chips[0][0]
            dq_ext = chips[0][1]
            d, g, rn, msk, _, xo, yo, _, _, _ = load_image(
                args.image, ext=sci_ext, dq_ext=dq_ext,
                dq_flags=args.dq_flags, lib_dir=args.lib_dir)
            return d, g, rn, msk, xo, yo

        _diag_data, _diag_gain, _diag_rn, _diag_mask, _xo, _yo = _load_diag_data(_diag_chip_key)

        def _chip_records(chip_key):
            recs = [r for r in stars if getattr(r, '_chip_ext', 1) == chip_key]
            return recs if recs else stars

        # Primary diagnostic plot
        if args.diagnostics:
            print(f"Generating diagnostic plot (q): {args.diagnostics}")
            plot_diagnostics(
                _chip_records(_diag_chip_key), _diag_data,
                psf_cube, xs, ys, psf_scale,
                hw=args.half_width, output=args.diagnostics,
                title=os.path.basename(args.image),
                residual=residuals.get(_diag_chip_key),
                mask=_diag_mask, noise_map=var_maps.get(_diag_chip_key))

        if args.diagnostics_Q:
            print(f"Generating diagnostic plot (Q): {args.diagnostics_Q}")
            plot_diagnostics_Q(
                _chip_records(_diag_chip_key), _diag_data,
                psf_cube, xs, ys, psf_scale,
                hw=args.half_width, output=args.diagnostics_Q,
                title=os.path.basename(args.image),
                residual=residuals.get(_diag_chip_key),
                mask=_diag_mask, noise_map=var_maps.get(_diag_chip_key))

        # Catalog stats plot (all chips combined)
        if args.catalog_stats:
            print(f"Generating catalog stats (q): {args.catalog_stats}")
            plot_catalog_stats(stars, output=args.catalog_stats,
                               title=os.path.basename(args.image))

        if args.catalog_stats_Q:
            print(f"Generating catalog stats (Q): {args.catalog_stats_Q}")
            plot_catalog_stats_Q(stars, output=args.catalog_stats_Q,
                                 title=os.path.basename(args.image))

        # PSF residual map (first chip)
        if args.psf_residual_map:
            print(f"Generating PSF residual map (q): {args.psf_residual_map}")
            plot_psf_residual_map(
                _chip_records(_diag_chip_key), _diag_data,
                psf_cube, xs, ys, psf_scale,
                hw=args.half_width, output=args.psf_residual_map,
                gain=_diag_gain, read_noise=_diag_rn,
                noise_map=var_maps.get(_diag_chip_key),
                mask=_diag_mask,
                title=os.path.basename(args.image))

        if args.psf_residual_map_Q:
            print(f"Generating PSF residual map (Q): {args.psf_residual_map_Q}")
            plot_psf_residual_map_Q(
                _chip_records(_diag_chip_key), _diag_data,
                psf_cube, xs, ys, psf_scale,
                hw=args.half_width, output=args.psf_residual_map_Q,
                gain=_diag_gain, read_noise=_diag_rn,
                noise_map=var_maps.get(_diag_chip_key),
                mask=_diag_mask,
                title=os.path.basename(args.image))

    print("Done.")

if __name__ == "__main__":
    main()
