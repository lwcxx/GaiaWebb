"""Tests for PSF loading and catalog output in jwst1pass."""

import numpy as np
import pytest
import os

from jwst1pass.io import load_psf_cube, catalog_to_table
from jwst1pass.core import StarRecord


# ---------------------------------------------------------------------------
# 1. Catalogue columns and dtypes
# ---------------------------------------------------------------------------

def test_catalog_columns():
    """catalog_to_table produces all expected columns with correct dtypes."""
    cov = np.eye(4) * 0.01
    records = [
        StarRecord(x=10.1, y=20.2, flux=1000.0, flux_err=10.0,
                   sky=200.0, sky_err=2.0, mag=15.0, mag_err=0.01,
                   qfit=0.05, chi2=1.1, central_res=0.02,
                   n_sat=0, psf_frac=0.27, psf_peak=0.28,
                   peak=500.0, cov=cov, pass_number=1,
                   n_neighbors=1, dist_nearest=40.2, dist_nearest_brighter=np.inf,
                   n_iter=5, converged=True, delta_max=0.001, chi2_scale=1.5),
    ]

    table = catalog_to_table(records, zero_point=25.0)

    required = ['x', 'y', 'flux', 'flux_err', 'sky', 'sky_err', 'mag', 'mag_err',
                'qfit', 'chi2', 'pass_number', 'n_iter', 'converged',
                'n_sat', 'psf_frac', 'psf_peak', 'peak',
                'n_neighbors', 'dist_nearest', 'dist_nearest_brighter',
                'cov_ff', 'cov_xx', 'cov_yy', 'cov_ss',
                'cov_fx', 'cov_fy', 'cov_fs', 'cov_xy', 'cov_xs', 'cov_ys',
                'chi2_scale']
    for col in required:
        assert col in table.colnames, f"Missing column: {col}"

    assert table['pass_number'].dtype.kind in ('i', 'u'), "pass_number should be integer"
    assert len(table) == 1
    assert table.meta.get('ZP') == 25.0


def test_catalog_empty():
    """catalog_to_table handles an empty record list."""
    table = catalog_to_table([], zero_point=25.0)
    assert len(table) == 0
    for col in ['x', 'y', 'flux', 'pass_number']:
        assert col in table.colnames
