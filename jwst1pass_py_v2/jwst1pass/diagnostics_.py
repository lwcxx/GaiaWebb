"""Diagnostic plots and summary statistics for jwst1pass catalogues."""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summarize_catalog(records, verbose=True, hw=3):
    """Compute and print summary statistics for a list of StarRecord.

    Parameters
    ----------
    records : list of StarRecord
    verbose : bool — if True, print a formatted summary to stdout
    hw      : int  — fit half-width used; crowding threshold = hw pixels

    Returns
    -------
    dict with summary statistics (always returned regardless of verbose).
    """
    n = len(records)
    if n == 0:
        if verbose:
            print("Catalog is empty.")
        return {'n_stars': 0}

    qfit   = np.array([r.qfit  for r in records])
    chi2   = np.array([r.chi2  for r in records])
    mag    = np.array([r.mag   for r in records])
    flux   = np.array([r.flux  for r in records])
    ferr   = np.array([r.flux_err for r in records])
    snr    = np.where(np.isfinite(ferr) & (ferr > 0), flux / ferr, 0.0)

    finite_q = qfit[np.isfinite(qfit)]
    finite_c = chi2[np.isfinite(chi2)]
    finite_m = mag[np.isfinite(mag)]

    n_sat_any  = sum(r.n_sat > 0 for r in records)
    n_crowded  = sum(r.dist_nearest_brighter < hw for r in records)
    n_conv     = sum(getattr(r, 'converged', True) for r in records)

    passes = sorted({r.pass_number for r in records})
    per_pass = {p: sum(1 for r in records if r.pass_number == p) for p in passes}

    stats = {
        'n_stars':      n,
        'qfit_median':  float(np.median(finite_q)) if len(finite_q) else np.nan,
        'qfit_p90':     float(np.percentile(finite_q, 90)) if len(finite_q) else np.nan,
        'chi2_median':  float(np.median(finite_c)) if len(finite_c) else np.nan,
        'chi2_p90':     float(np.percentile(finite_c, 90)) if len(finite_c) else np.nan,
        'snr_median':   float(np.median(snr)),
        'mag_min':      float(np.min(finite_m)) if len(finite_m) else np.nan,
        'mag_max':      float(np.max(finite_m)) if len(finite_m) else np.nan,
        'n_sat_any':    n_sat_any,
        'frac_crowded': n_crowded / n,
        'n_converged':  n_conv,
        'per_pass':     per_pass,
    }

    if verbose:
        sep = '=' * 52
        print(f"\n{sep}")
        print(f"  Catalog summary  ({n} stars)")
        print(sep)
        if len(finite_m):
            print(f"  Magnitude range : {stats['mag_min']:.2f} → {stats['mag_max']:.2f}")
        print(f"  Pass breakdown  : " +
              ", ".join(f"pass {p}: {c}" for p, c in per_pass.items()))
        print(f"  qfit  : median={stats['qfit_median']:.3f}  "
              f"90th={stats['qfit_p90']:.3f}   (>0.5 = poor)")
        print(f"  chi²  : median={stats['chi2_median']:.3f}  "
              f"90th={stats['chi2_p90']:.3f}   (~1.0 = ideal)")
        print(f"  S/N   : median={stats['snr_median']:.1f}")
        print(f"  Saturated (n_sat > 0)        : "
              f"{n_sat_any} ({100*n_sat_any/n:.1f}%)")
        print(f"  Crowded (brighter nbr <{hw}px) : "
              f"{n_crowded} ({100*n_crowded/n:.1f}%)")
        n_nc = n - n_conv
        if n_nc:
            print(f"  Did not converge (max_iter)   : "
                  f"{n_nc} ({100*n_nc/n:.1f}%)")
        print(sep)

    return stats


# ---------------------------------------------------------------------------
# Catalog statistics plot
# ---------------------------------------------------------------------------

def plot_catalog_stats(records, output=None, title=None):
    """Statistical diagnostic plots for a jwst1pass catalog.

    Creates a multi-panel figure with:
      Row 1 : σ_x vs mag,  σ_y vs mag
      Row 2 : qfit vs mag,  chi² vs mag
      Row 3 : chi² histogram
      Row 4 : x pixel-phase vs x,  y pixel-phase vs y
      Row 5 : 2-D scatter of (x_phase, y_phase)

    Parameters
    ----------
    records : list of StarRecord
    output  : file path to save figure (None → don't save)
    title   : optional figure suptitle

    Returns
    -------
    matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError("matplotlib is required for plot_catalog_stats.")

    if not records:
        raise ValueError("No records to plot.")

    mag        = np.array([r.mag        for r in records])
    flux       = np.array([r.flux       for r in records])
    sx         = np.array([np.sqrt(max(r.cov[1,1], 0.0)) if r.cov is not None else np.nan for r in records])
    sy         = np.array([np.sqrt(max(r.cov[2,2], 0.0)) if r.cov is not None else np.nan for r in records])
    q          = np.array([r.qfit       for r in records])
    chi2       = np.array([r.chi2       for r in records])
    chi2_scale = np.array([getattr(r, 'chi2_scale', 1.0) for r in records])
    x          = np.array([r.x          for r in records])
    y          = np.array([r.y          for r in records])
    conv       = np.array([r.converged  for r in records])
    n_iter     = np.array([r.n_iter     for r in records])
    n_sat      = np.array([r.n_sat      for r in records])
    xph        = x % 1.0
    yph        = y % 1.0

    finite = np.isfinite(mag) & np.isfinite(sx) & np.isfinite(sy) \
             & np.isfinite(chi2) & np.isfinite(q)
    keep = (q < 2) & conv & (chi2 < 10) & (flux > 1.1)
    finite &= keep
    mag_f  = mag[finite];  sx_f  = sx[finite];   sy_f  = sy[finite]
    q_f    = q[finite];    chi2_f = chi2[finite]
    chi2_scale_f = chi2_scale[finite]
    x_f, y_f, xph_f, yph_f = x[finite], y[finite], xph[finite], yph[finite]
    n_iter_f = n_iter[finite]

    # chi²_scale running-median curve (used in two panels)
    def _chi2_scale_curve(mag_vals, cs_vals, n_pts=200):
        """Return (mag_curve, cs_curve) smooth running-median for overlay."""
        if mag_vals.size < 10:
            return np.array([]), np.array([])
        order = np.argsort(mag_vals)
        ms = mag_vals[order];  cs = cs_vals[order]
        bin_size = max(6, len(ms) // 40)
        bm, bc = [], []
        for i in range(0, len(ms), bin_size):
            chunk = slice(i, min(i + bin_size, len(ms)))
            bm.append(float(np.median(ms[chunk])))
            bc.append(float(np.median(cs[chunk])))
        return np.array(bm), np.array(bc)

    fig = plt.figure(figsize=(14, 22), layout='constrained')
    gs  = gridspec.GridSpec(6, 2, figure=fig)

    # Colour by iteration count
    unique_iters = sorted(set(n_iter_f.tolist()))
    cmap = plt.get_cmap('viridis')
    vmax_iter = max(unique_iters) if unique_iters else 15

    def _scatter_by_iter(ax, xdata, ydata, **kw):
        return ax.scatter(xdata, ydata, c=n_iter_f, cmap=cmap, vmin=1, vmax=vmax_iter, **kw)

    kw_sc = dict(s=5, alpha=0.4, rasterized=True, edgecolors='none')
    _mag_lim = (np.nanmin(mag_f) - 0.5, np.nanmax(mag_f) + 0.5) if mag_f.size else (0, 1)

    # --- Row 0: position uncertainties vs mag --------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_by_iter(ax1, mag_f, sx_f, **kw_sc)
    ax1.set_xlabel('Magnitude')
    ax1.set_ylabel('σ_x  (pix)')
    ax1.set_title('X position uncertainty vs mag')
    ax1.set_xlim(_mag_lim)
    ax1.set_yscale('log')
    ax1.set_ylim(5e-4, 5)

    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_by_iter(ax2, mag_f, sy_f, **kw_sc)
    ax2.set_xlabel('Magnitude')
    ax2.set_ylabel('σ_y  (pix)')
    ax2.set_title('Y position uncertainty vs mag')
    ax2.set_xlim(_mag_lim)
    ax2.set_yscale('log')
    ax2.set_ylim(5e-4, 5)

    # --- Row 1: qfit and raw chi² -------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _scatter_by_iter(ax3, mag_f, q_f, **kw_sc)
    ax3.axhline(0.1, color='green', lw=0.8, ls='--', label='excellent (0.1)')
    ax3.axhline(0.5, color='red',   lw=0.8, ls='--', label='poor (0.5)')
    ax3.set_xlabel('Magnitude')
    ax3.set_ylabel('qfit')
    ax3.set_title('Quality of fit vs mag')
    ax3.set_yscale('log')
    ax3.set_xlim(_mag_lim)

    ax4 = fig.add_subplot(gs[1, 1])
    _scatter_by_iter(ax4, mag_f, chi2_f, **kw_sc)
    _bm, _bc = _chi2_scale_curve(mag_f, chi2_scale_f)
    if _bm.size:
        ax4.plot(_bm, _bc, 'r-', lw=1.8, zorder=5, label='chi²_scale curve')
    ax4.axhline(1.0, color='green', lw=0.8, ls='--', label='ideal (1.0)')
    ax4.set_xlabel('Magnitude')
    ax4.set_ylabel('chi²')
    ax4.set_title('Raw chi² vs mag  (red = correction curve)')
    ax4.set_xlim(_mag_lim)
    _chi2_fin = chi2_f[np.isfinite(chi2_f)]
    _chi2_top = float(np.percentile(_chi2_fin, 99)) if _chi2_fin.size else 5.0
    ax4.set_ylim(0, min(_chi2_top * 1.5, 10.0))

    # --- Row 2: chi²_scale vs mag and chi² histogram -------------------------
    ax5 = fig.add_subplot(gs[2, 0])
    _scatter_by_iter(ax5, mag_f, chi2_scale_f, **kw_sc)
    if _bm.size:
        ax5.plot(_bm, _bc, 'r-', lw=1.8, zorder=5)
    ax5.axhline(1.0, color='green', lw=0.8, ls='--', label='no scaling (1.0)')
    ax5.set_xlabel('Magnitude')
    ax5.set_ylabel('chi²_scale')
    ax5.set_title('Applied chi²-scaling vs mag')
    ax5.set_xlim(_mag_lim)
    _cs_fin = chi2_scale_f[np.isfinite(chi2_scale_f)]
    if _cs_fin.size:
        _cs_lo = max(float(np.percentile(_cs_fin, 1))  * 0.9, 0.1)
        _cs_hi = min(float(np.percentile(_cs_fin, 99)) * 1.1, 10.0)
        ax5.set_ylim(_cs_lo, _cs_hi)

    ax6 = fig.add_subplot(gs[2, 1])
    _chi2_plot = chi2_f[np.isfinite(chi2_f) & (chi2_f < 10)]
    if _chi2_plot.size:
        ax6.hist(_chi2_plot, bins=80, color='steelblue', alpha=0.7,
                 density=True, label='observed')
        _chi2_med = float(np.median(_chi2_plot))
        _cs_med   = float(np.median(_cs_fin)) if _cs_fin.size else 1.0
        ax6.axvline(1.0,      color='green', lw=1.0, ls='--', label='ideal (1.0)')
        ax6.axvline(_chi2_med, color='navy', lw=1.0, ls=':',
                    label=f'raw median ({_chi2_med:.2f})')
        ax6.axvline(_cs_med,   color='red',  lw=1.0, ls='-.',
                    label=f'scale median ({_cs_med:.2f})')
    ax6.set_xlabel('Reduced chi²')
    ax6.set_ylabel('Density')
    ax6.set_title('chi² distribution')
    ax6.legend(fontsize=7)

    # --- Row 3: chi²_scale vs chi² (residual check) --------------------------
    ax7 = fig.add_subplot(gs[3, :])
    _ratio = chi2_f / np.where(chi2_scale_f > 0, chi2_scale_f, 1.0)
    _scatter_by_iter(ax7, mag_f, _ratio, **kw_sc)
    ax7.axhline(1.0, color='red', lw=1.0, ls='--', label='ratio = 1 (ideal)')
    ax7.set_xlabel('Magnitude')
    ax7.set_ylabel('raw chi² / chi²_scale')
    ax7.set_title('Residual after scaling  (should be flat at 1)')
    ax7.set_xlim(_mag_lim)
    _r_fin = _ratio[np.isfinite(_ratio)]
    if _r_fin.size:
        ax7.set_ylim(max(0, float(np.percentile(_r_fin, 1)) * 0.8),
                     min(float(np.percentile(_r_fin, 99)) * 1.2, 5.0))
    ax7.legend(fontsize=7, markerscale=3)

    # Colorbar for iteration count — spans all scatter panels for visibility
    _iter_sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=vmax_iter))
    _iter_sm.set_array([])
    _iter_cbar = fig.colorbar(_iter_sm, ax=[ax1, ax2, ax3, ax4, ax5, ax7],
                               location='right', fraction=0.02, pad=0.01, shrink=0.95)
    _iter_cbar.set_label('Solver iterations', fontsize=10)
    if vmax_iter <= 20:
        _iter_cbar.set_ticks(list(range(1, vmax_iter + 1)))

    # --- Rows 4–5: 2-D pixel phase scatter -----------------------------------
    ax8 = fig.add_subplot(gs[4:, :])
    if xph_f.size > 500:
        h, xe, ye = np.histogram2d(xph_f, yph_f, bins=50,
                                    range=[[0, 1], [0, 1]])
        im = ax8.imshow(h.T, origin='lower', extent=[0, 1, 0, 1],
                        aspect='equal', cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=ax8, fraction=0.02, pad=0.01, label='Count')
        ax8.set_title('2-D pixel phase distribution  (2D histogram)')
    else:
        ax8.scatter(xph_f, yph_f, c=n_iter_f, cmap=cmap, vmin=1, vmax=vmax_iter, **kw_sc)
        ax8.set_title('2-D pixel phase distribution')
    ax8.set_xlabel('X phase  (x mod 1)')
    ax8.set_ylabel('Y phase  (y mod 1)')
    ax8.set_xlim(-0.02, 1.02)
    ax8.set_ylim(-0.02, 1.02)
    ax8.set_aspect('equal')

    if title:
        fig.suptitle(title, fontsize=11)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# PSF residual map
# ---------------------------------------------------------------------------

def plot_psf_residual_map(records, data, psf_cube, xs, ys, psf_scale, hw,
                           n_grid=4, output=None, title=None,
                           min_stars=5, gain=1.0, read_noise=5.0,
                           noise_map=None, mask=None):
    """Show average fractional PSF residuals across a spatial grid of the detector."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ImportError("matplotlib is required for plot_psf_residual_map.")

    from scipy.ndimage import spline_filter
    from .core import _eval_psf_grad_fast, _window_offsets
    from .utils import interpolate_psf

    ny, nx = data.shape
    win = 2 * hw + 1

    psf_coeffs_cube = np.array([
        spline_filter(p, order=3, output=np.float64) for p in psf_cube
    ])

    x_edges = np.linspace(0, nx, n_grid + 1)
    y_edges = np.linspace(0, ny, n_grid + 1)

    sum_wr  = np.zeros((n_grid, n_grid, win, win))
    sum_w   = np.zeros((n_grid, n_grid, win, win))
    n_stars = np.zeros((n_grid, n_grid), dtype=int)

    good_recs = [r for r in records
                 if r.qfit < 0.1 and getattr(r, 'converged', True) and r.chi2 < 4
                 and r.flux > 1.1 and np.isfinite(r.mag)]

    for rec in good_recs:
        ix_tile = int(np.clip(np.searchsorted(x_edges[1:], rec.x), 0, n_grid - 1))
        iy_tile = int(np.clip(np.searchsorted(y_edges[1:], rec.y), 0, n_grid - 1))

        xi = int(round(rec.x)); yi = int(round(rec.y))
        dx = rec.x - xi;        dy = rec.y - yi

        y_lo, y_hi, x_lo, x_hi, diy, dix = _window_offsets(xi, yi, hw, ny, nx)
        if (y_hi - y_lo) != win or (x_hi - x_lo) != win:
            continue

        d_stamp = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)

        local_psf = interpolate_psf(psf_coeffs_cube, xs, ys, rec.x, rec.y)
        P, _, _ = _eval_psf_grad_fast(local_psf, dx, dy, dix, diy, psf_scale)

        r_stamp = d_stamp - rec.sky - rec.flux * P

        if noise_map is not None:
            var_stamp = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        else:
            var_stamp = (np.maximum(rec.flux * P, 0.0) + max(rec.sky, 0.0)) / gain \
                        + (read_noise / gain) ** 2
        var_stamp = np.maximum(var_stamp, 1e-10)

        norm = max(rec.flux * rec.psf_peak, 1e-10)
        r_norm = r_stamp / norm

        clip_m = getattr(rec, 'clipped_mask', None)
        bad = ~np.isfinite(r_norm)
        if clip_m is not None and clip_m.shape == bad.shape:
            bad |= (clip_m != 0)
        if mask is not None:
            bad |= mask[y_lo:y_hi, x_lo:x_hi].astype(bool)

        w = np.where(bad, 0.0, 1.0 / var_stamp)
        
        # Avoid NaN propagation: 0.0 * NaN = NaN in numpy. 
        # Explicitly zero out the weighted residual for bad pixels.
        wr = np.zeros_like(r_norm)
        wr[~bad] = w[~bad] * r_norm[~bad]

        sum_wr[iy_tile, ix_tile] += wr
        sum_w [iy_tile, ix_tile] += w
        n_stars[iy_tile, ix_tile] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        avg_r = np.where(sum_w > 0, sum_wr / sum_w, np.nan)

    fig, axes = plt.subplots(n_grid, n_grid, figsize=(n_grid * 2.2, n_grid * 2.2),
                             layout='constrained')
    if n_grid == 1:
        axes = np.array([[axes]])

    vals = avg_r[np.isfinite(avg_r)]
    vlim = float(np.percentile(np.abs(vals), 98)) if vals.size else 0.05
    vlim = max(vlim, 1e-6)
    norm_r = Normalize(vmin=-vlim, vmax=vlim)

    kw_im = dict(origin='lower', interpolation='nearest', aspect='equal',
                 cmap='RdBu_r', norm=norm_r)

    for iy in range(n_grid):
        for ix in range(n_grid):
            ax = axes[n_grid - 1 - iy, ix]
            ax.set_facecolor('#f0f0f0') # Light gray background for all tiles
            n = n_stars[iy, ix]
            tile_r = avg_r[iy, ix]
            if n >= min_stars and np.isfinite(tile_r).any():
                im = ax.imshow(tile_r, **kw_im)
                ax.set_title(f'N={n}', fontsize=7, pad=1)
            else:
                ax.set_facecolor('#cccccc')
                ax.text(0.5, 0.5, f'N={n}\n(insuf.)', ha='center', va='center',
                        fontsize=7, transform=ax.transAxes, color='#555555')
            ax.add_patch(Rectangle((-0.5, -0.5), win, win,
                                   fill=False, edgecolor='cyan',
                                   linewidth=0.5, linestyle='--'))
            ax.set_xticks([]); ax.set_yticks([])

    sm = plt.cm.ScalarMappable(norm=norm_r, cmap='RdBu_r')
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02,
                 label='Fractional PSF residual  (data−model) / (flux·PSF_peak)')

    if title:
        fig.suptitle(title, fontsize=10)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Postage stamp diagnostic plot
# ---------------------------------------------------------------------------

def plot_diagnostics(records, data, psf_cube, xs, ys, psf_scale, hw,
                     output=None, n_stamps=16, title=None,
                     stamp_pad=2, cmap_data='gray', cmap_res='RdBu_r',
                     residual=None, mask=None, noise_map=None):
    """Create a postage-stamp diagnostic figure: data / PSF model / residual."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, SymLogNorm
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ImportError("matplotlib is required for plot_diagnostics.")

    from scipy.ndimage import spline_filter
    from .core import _eval_psf_grad_fast
    from .utils import interpolate_psf

    if not records:
        raise ValueError("No records to plot.")

    valid = [r for r in records
             if np.isfinite(r.mag) and np.isfinite(r.qfit)
             and r.flux > 1.1 and r.chi2 < 4 and getattr(r, 'converged', True)]
    if not valid:
        raise ValueError("No valid records.")

    good_fits = sorted([r for r in valid if r.qfit < 0.1], key=lambda r: r.mag)
    poor_fits = sorted([r for r in valid if r.qfit >= 0.1], key=lambda r: r.mag)

    n_poor = min(len(poor_fits), max(1, n_stamps // 5))
    n_good = min(len(good_fits), n_stamps - n_poor)
    n_poor = n_stamps - n_good

    def _uniform_sample(lst, k):
        if not lst or k <= 0: return []
        if k >= len(lst): return list(lst)
        idx = np.linspace(0, len(lst) - 1, k).round().astype(int)
        return [lst[i] for i in idx]

    selected = _uniform_sample(good_fits, n_good) + _uniform_sample(poor_fits, n_poor)
    selected.sort(key=lambda r: r.mag)
    n_sel = len(selected)

    _NCOLS_PER_STAR = 4
    n_col_groups = max(1, min(3, int(np.ceil(np.sqrt(n_sel / 1.5)))))
    n_rows = int(np.ceil(n_sel / n_col_groups))
    n_cols = n_col_groups * _NCOLS_PER_STAR

    stamp_size = 2 * (hw + stamp_pad) + 1
    inch_per_stamp = max(1.6, stamp_size / 8.0)
    fig_w = n_cols * inch_per_stamp
    fig_h = n_rows * inch_per_stamp + (0.5 if title else 0.1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False, layout='constrained')

    psf_coeffs_cube = np.array([
        spline_filter(p, order=3, output=np.float64) for p in psf_cube
    ])
    ny, nx = data.shape

    for idx, rec in enumerate(selected):
        row = idx // n_col_groups
        cg  = idx %  n_col_groups
        cb  = cg * _NCOLS_PER_STAR
        ax_d, ax_m, ax_r, ax_s = (axes[row, cb], axes[row, cb+1],
                                   axes[row, cb+2], axes[row, cb+3])
        
        # Use light gray background to distinguish masked pixels (white = 0 residual)
        for ax in (ax_d, ax_m, ax_r, ax_s):
            ax.set_facecolor('#f0f0f0')

        xi = int(round(rec.x)); yi = int(round(rec.y))
        shw = hw + stamp_pad
        y_lo = max(0, yi - shw); y_hi = min(ny, yi + shw + 1)
        x_lo = max(0, xi - shw); x_hi = min(nx, xi + shw + 1)

        dx = rec.x - xi; dy = rec.y - yi
        diy_s = (np.arange(y_lo, y_hi) - yi)[:, np.newaxis]
        dix_s = (np.arange(x_lo, x_hi) - xi)[np.newaxis, :]
        local_psf = interpolate_psf(psf_coeffs_cube, xs, ys, rec.x, rec.y)
        P_s, _, _ = _eval_psf_grad_fast(local_psf, dx, dy, dix_s, diy_s, psf_scale)
        stamp_m = rec.flux * P_s + rec.sky

        if residual is not None:
            stamp_d = (residual[y_lo:y_hi, x_lo:x_hi] + rec.flux * P_s).astype(np.float64)
        else:
            stamp_d = data[y_lo:y_hi, x_lo:x_hi].astype(np.float64)

        stamp_r = stamp_d - stamp_m

        # Per-pixel noise for residual/sigma panel.
        if noise_map is not None:
            var_s = noise_map[y_lo:y_hi, x_lo:x_hi].astype(np.float64)
        else:
            # Replicate core.py noise model: (flux*P + sky)/gain + (rn/gain)^2
            # Use gain=1.0, rn=10.0 as defaults if not provided (approx for MIRI/NIRISS)
            var_s = (np.maximum(stamp_m, 0.0)) / 1.0 + (10.0 / 1.0)**2
        var_s = np.maximum(var_s, 1e-10)
        stamp_rs = stamp_r / np.sqrt(var_s)

        if mask is not None:
            # Only mask pixels marked as DO_NOT_USE (Bit 0) in residual panels
            bad = (mask[y_lo:y_hi, x_lo:x_hi] & 1) != 0
            stamp_r  = np.where(bad, np.nan, stamp_r)
            stamp_rs = np.where(bad, np.nan, stamp_rs)
        
        above_d = stamp_d - max(rec.sky, 1.0)
        above_m = stamp_m - max(rec.sky, 1.0)

        # Apply clipping mask (from fit_star)
        clip_m = getattr(rec, 'clipped_mask', None)
        if clip_m is not None and clip_m.shape == stamp_r.shape:
            # NaN out clipped pixels in residuals so they don't bias the scale
            stamp_r  = np.where(clip_m != 0, np.nan, stamp_r)
            stamp_rs = np.where(clip_m != 0, np.nan, stamp_rs)
            # Keep them in the Data panel to show context, but NaN out DQ Bit 0
            if mask is not None:
                above_d = np.where((mask[y_lo:y_hi, x_lo:x_hi] & 1) != 0, np.nan, above_d)

        peak_signal = max(np.nanmax(above_d) if np.any(np.isfinite(above_d)) else 1.0, 
                          np.max(above_m), 1.0)
        linthresh = max(0.01 * peak_signal, 1.0)
        norm_log = SymLogNorm(linthresh=linthresh, vmin=-linthresh, vmax=peak_signal, base=10)

        r_valid  = stamp_r[np.isfinite(stamp_r)]
        rs_valid = stamp_rs[np.isfinite(stamp_rs)]
        res_lim  = max(np.abs(r_valid).max(),  1e-10) if r_valid.size  else 1.0
        rs_lim   = max(np.abs(rs_valid).max(), 1.0)   if rs_valid.size else 5.0
        rs_lim   = np.clip(rs_lim, 5.0, 20.0)
        norm_r  = Normalize(vmin=-res_lim, vmax=res_lim)
        norm_rs = Normalize(vmin=-rs_lim,  vmax=rs_lim)

        kw_im = dict(origin='lower', interpolation='nearest', aspect='equal')
        im_d = ax_d.imshow(above_d, norm=norm_log, cmap=cmap_data, **kw_im)
        im_m = ax_m.imshow(above_m, norm=norm_log, cmap=cmap_data, **kw_im)
        im_r = ax_r.imshow(stamp_r,  norm=norm_r,   cmap=cmap_res,  **kw_im)
        im_s = ax_s.imshow(stamp_rs, norm=norm_rs,  cmap=cmap_res,  **kw_im)

        _cb_kw = dict(fraction=0.046, pad=0.04, aspect=12)
        fig.colorbar(im_r, ax=ax_r, **_cb_kw)
        cbar_s = fig.colorbar(im_s, ax=ax_s, **_cb_kw)
        cbar_s.set_label('σ', fontsize=5, labelpad=2)

        for ax in (ax_d, ax_m, ax_r, ax_s):
            ax.set_xticks([]); ax.set_yticks([])
            fw_x0 = (xi - hw) - x_lo - 0.5
            fw_y0 = (yi - hw) - y_lo - 0.5
            fw_w  = 2 * hw + 1
            ax.add_patch(Rectangle((fw_x0, fw_y0), fw_w, fw_w, fill=False, 
                                   edgecolor='cyan', linewidth=0.6, linestyle='--'))

        conv_tag = "" if getattr(rec, 'converged', True) else " !"
        label = f"m={rec.mag:.2f}  q={rec.qfit:.3f}{conv_tag}\n({rec.x:.0f},{rec.y:.0f}) P{rec.pass_number}"
        ax_d.text(0.03, 0.97, label, transform=ax_d.transAxes, fontsize=5, color='white',
                  va='top', ha='left', bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55))

    for cg in range(n_col_groups):
        cb = cg * _NCOLS_PER_STAR
        if n_rows > 0:
            axes[0, cb    ].set_title('Data (log)',    fontsize=7, pad=2)
            axes[0, cb + 1].set_title('Model (log)',   fontsize=7, pad=2)
            axes[0, cb + 2].set_title('Residual',      fontsize=7, pad=2)
            axes[0, cb + 3].set_title('Residual / σ',  fontsize=7, pad=2)

    for idx in range(n_sel, n_rows * n_col_groups):
        row, cg = idx // n_col_groups, idx % n_col_groups
        cb = cg * _NCOLS_PER_STAR
        for dc in range(_NCOLS_PER_STAR): axes[row, cb + dc].set_visible(False)

    if title: fig.suptitle(title, fontsize=9)
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def save_standard_plots(records, data, psf_cube, xs, ys, psf_scale, hw=5,
                        image_path=None, out_dir=None, base_name=None,
                        subfolder='plots', residual=None, mask=None, 
                        noise_map=None, verbose=True):
    """
    Convenience function to generate and save all standard diagnostic plots.

    Parameters
    ----------
    records : list of StarRecord
        The catalog of stars to plot.
    data : 2D array
        The science image data.
    psf_cube, xs, ys, psf_scale : PSF model components
    hw : int
        Fit half-width used.
    image_path : str, optional
        Path to the input FITS file. Used to derive out_dir and base_name if not provided.
    out_dir : str, optional
        Main directory for output. Defaults to image_path's directory or '.'.
    base_name : str, optional
        Base filename for the plots. Defaults to image_path's basename or 'jwst1pass'.
    subfolder : str, optional
        Sub-directory within out_dir for plots. Defaults to 'plots'. 
        Set to None or '' to save directly in out_dir.
    residual : 2D array, optional
        Residual image for star stamps.
    mask : 2D array, optional
        DQ mask for masking bad pixels in stamps.
    noise_map : 2D array, optional
        Pixel-level noise map.
    verbose : bool
        If True, print saved file paths.
    """
    if not records:
        return

    # Derive output paths
    if image_path:
        if out_dir is None:
            out_dir = os.path.dirname(image_path) or '.'
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if out_dir is None:
        out_dir = '.'
    if base_name is None:
        base_name = 'jwst1pass'

    plot_dir = out_dir
    if subfolder:
        plot_dir = os.path.join(out_dir, subfolder)

    # Ensure output directory exists
    if plot_dir != '.' and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    # 1. Catalog stats
    stats_out = os.path.join(plot_dir, f"{base_name}_stats.png")
    if verbose: print(f"  -> {stats_out}")
    plot_catalog_stats(records, output=stats_out, 
                       title=f"jwst1pass stats: {os.path.basename(image_path or base_name)}")

    # 2. PSF residual map
    resid_out = os.path.join(plot_dir, f"{base_name}_psf_resid.png")
    if verbose: print(f"  -> {resid_out}")
    plot_psf_residual_map(records, data, psf_cube, xs, ys, psf_scale, hw=hw,
                          output=resid_out,
                          title=f"PSF residuals: {os.path.basename(image_path or base_name)}")

    # 3. Star stamps
    stamps_out = os.path.join(plot_dir, f"{base_name}_stamps.png")
    if verbose: print(f"  -> {stamps_out}")
    plot_diagnostics(records, data, psf_cube, xs, ys, psf_scale, hw=hw,
                     output=stamps_out, residual=residual, mask=mask,
                     noise_map=noise_map,
                     title=f"Star stamps: {os.path.basename(image_path or base_name)}")
