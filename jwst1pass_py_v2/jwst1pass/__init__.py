"""jwst1pass — Python PSF-fitting photometry for JWST images (MIRI/NIRISS)"""

from .core import fit_star, find_sources, StarRecord, inflate_chi2, estimate_sky, _build_chi2_mag_bins
from .multipass import run_photometry, subtract_stars, build_variance_image, refit_stars
from .io import load_image, load_psf_cube, load_stdgdc, apply_gdc, catalog_to_table, run_photometry_fits
from .diagnostics import (summarize_catalog, plot_diagnostics,
                           plot_catalog_stats, plot_psf_residual_map,
                           plot_concentration_diagnostics,
                           estimate_systematic_floor,
                           measure_psf_perturbation,
                           plot_psf_perturbation,
                           save_standard_plots)

__version__ = "0.1.0"
