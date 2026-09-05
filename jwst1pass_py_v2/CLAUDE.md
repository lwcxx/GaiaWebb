# GaiaWebb: High-Precision Astrometry and Photometry

This repository contains tools for processing astronomical images from the Hubble Space Telescope (HST) and James Webb Space Telescope (JWST), integrated with Gaia mission data.

## jwst1pass

Modern Python implementation for JWST PSF-fitting photometry (MIRI, NIRISS).

### Key Features
- **Multi-pass Photometry**: Iterative discovery and measurement to handle crowded fields and faint neighbours.
- **Robust Fitting**: Joint 4-parameter solver (flux, x, y, sky) with **pre-solve initialization** (linear LS fit) and **safety clamps** on position steps and flux floors.
- **Seed Jittering**: Implements random sub-pixel jitter in `(-0.5, 0.5)` for discovery seeds to break pixel-phase bias, significantly improving centroid convergence and astrometric precision.
- **Poisson Noise Model**: Star-aware variance maps that account for the Poisson noise of all detected sources, improving fit quality in dense regions.
- **Advanced Discovery**: High-speed `find_sources` algorithm using 8-neighbour local maxima and vectorized sigma-clipped sky estimation.
- **GDC & WCS**: Integrated geometric distortion correction and WCS sky-coordinate transformation with full error propagation.
- **Autonomous Metadata Generation**: Automatically produces `<image>_GDC_vals.csv` and `<image>_GDC_vals_REFCOORDS.csv` auxiliary files, replicating the legacy post-processing logic with high numerical precision (matching to 12 decimal places).
- **Diagnostics**: Comprehensive suite of diagnostic plots (catalog stats, PSF residuals, star stamps) with automated save drivers.

### Core Components
- `core.py`: Newton-loop PSF fitter (`fit_star`), vectorized sky estimation (`estimate_sky_batch`), discovery (`find_sources`), concentration metrics (computed inside `fit_star`), star/galaxy classification (`classify_stars`, `_conc_adaptive_bounds`), adaptive chi² binning (`_build_chi2_mag_bins`), and magnitude-dependent covariance rescaling (`inflate_chi2`).
- `multipass.py`: Star subtraction, leave-one-out re-fitting coordinator (`refit_stars`), and Poisson noise map generator (`build_variance_image`). Supports `n_discovery_passes` to limit discovery iterations.
- `io.py`: FITS loaders, GDC/WCS application, legacy metadata generation (`write_gdc_auxiliary_files`), and high-level `run_photometry_fits` driver for automated end-to-end processing.
- `diagnostics.py`: Statistical and visual diagnostic tools. Imports `_build_chi2_mag_bins` from `core.py` (shared with `inflate_chi2` to guarantee the chi² overlay curve matches what was applied). Includes `estimate_systematic_floor`, `plot_concentration_diagnostics`, `measure_psf_perturbation`, `plot_psf_perturbation`, and the `save_standard_plots` convenience function.

### Standards
- **Numerical Precision**: Use `float64` for all coordinate and flux calculations.
- **Performance**: Numba-accelerated B-spline kernels for high-speed PSF evaluation.
- **Safety**: Newton steps are clamped and monitored for divergence.
- **Types**: Use Python 3.10+ type hints and dataclasses for structured records.

## Project Structure

- **[jwst1pass_py/](./jwst1pass_py/)**: Modern JWST photometry suite featuring a Python reimplementation (`jwst1pass`). (Primary Documentation below)
- **[hst1pass_improved/](./hst1pass_improved/)**: Improved HST photometry suite featuring the `py1pass` package and updated Fortran sources.
- **[fortran_codes/](./fortran_codes/CLAUDE.md)**: Original Fortran-based tools for one-pass photometry and astrometric calibration (`hst1pass`, `jwst1pass`).
- **`python_codes/`**: Gaia integration and data cleaning scripts (e.g., `GaiaHub.py`, `GaiaJWST.py`).
- **`lib/`**: Shared library of Standard Distortion Corrections (STDGDCs) and Standard PSFs (STDPSFs).

## JWST Photometry Suite (`jwst1pass_py`)

### Internal Architecture (`jwst1pass`)

- **Joint Solver**: Uses a 4-parameter linear least-squares solver (flux, x, y, sky) per Newton iteration, replacing separate 1D steps for superior convergence in the undersampled JWST regime.
- **PSF Interpolation**: Cubic B-spline interpolation among spatial PSF grid models (`scipy.ndimage.spline_filter` prefilter + `map_coordinates(order=3)`, with a Numba-accelerated `_bspline3` fast path), ensuring accuracy for complex JWST PSF shapes and the cruciform artifact. Note: this is cubic B-spline (C² continuous, globally prefiltered), distinct from Catmull-Rom (C¹ Hermite) and from the Fortran engine's local finite-difference bicubic.
- **Geometric Distortion (GDC)**: Forward distortion and pixel-area magnitude corrections are applied via bilinear interpolation of image-based STDGDC maps.
- **Uncertainty Rescaling**: Covariances are rescaled by magnitude-dependent median chi² (`inflate_chi2` in `core.py`) to account for PSF model mismatch. `inflate_chi2` uses `_build_chi2_mag_bins` — the same adaptive binning and Gaussian smoothing function used by `plot_catalog_stats` to draw the chi² overlay curve — so the plotted curve always exactly represents what was applied to the covariances. 
    - **Noise Model**: All instruments (NIRISS, MIRI, NIRCam) operate in native **MJy/sr** units, matching the Fortran default (`XFACTOR=1`, no scaling). The per-pixel variance is:
        - **When `pipeline_noise_map` is provided** (ERR extension loaded): `var = ERR² + (0.01·flux·P)²`. The pipeline ERR already encodes stellar Poisson from ramp fitting, so no self-Poisson term is added — matching the HST `py1pass` design. Neighbor Poisson is accumulated separately by `build_variance_image` between passes.
        - **When `pipeline_noise_map` is absent**: `var = (flux·P + sky)/g_eff + (RN/g_eff)² + (0.01·flux·P)²` — full Poisson+RN model.
        - **Effective gain**: `g_eff = hardware_gain × xfactor` [e⁻/(MJy/sr)], where `xfactor = EFFEXPTM / PHOTMJSR`. For MIRI F560W this gives `g_eff ≈ 81 e⁻/(MJy/sr)`.
    - **Systematic Floor**: A 1% systematic noise floor `(0.01·flux·P)²` is always added, matching the Fortran `model_sig2 = (0.01·zu·psfx)²` term in `find_xyzXXe_NAXIS`.
    - **Root-Scale Convention**: To ensure statistical consistency with the improved HST suite, the `StarRecord.chi2` attribute stores the **square root of the reduced chi-squared** ($\sqrt{\chi^2_{red}}$), and the `chi2_scale` correction factor is likewise a **root-scale**.
    - **Maintenance & Stability**: 
        - **Error Inflation**: Fixed a critical bug where poor fits (`qfit >= 9.0`) were skipped during covariance scaling. These sources now correctly receive valid uncertainty estimates (e.g., `inf` or self-scaled).
        - **Numerical Safety**: All failed fits or high-qfit sources have their uncertainties explicitly flagged as `inf` in the output catalog to prevent incorrect interpretation of unstable results.
        - **NaN-Robustness**: Hardened the shared photometry engine against background corruption by transitioning to `np.nanmedian` and explicit `np.isfinite` filtering in both scalar and batch modes. This ensures both NIRISS and MIRI achieve numerical parity and robustness when stars are located near masked pixels or detector boundaries.
- **Unit Convention**: All instruments (NIRISS, MIRI, NIRCam) keep data in native **MJy/sr** units throughout photometry, matching the Fortran `jwst1pass` default (`XFACTOR=1`). No DN scaling is applied. The effective gain is scaled instead: `g_eff = hardware_gain × xfactor [e⁻/(MJy/sr)]`, so Poisson and read-noise variance terms remain in (MJy/sr)². This ensures `fmin` has the same physical meaning across all cameras.
- **Amplifier Gains**: 
    - **NIRISS**: The hardware gain defaults to **1.9835** (mean of amplifiers A, B, C, D) for final electron conversion.
    - **MIRI**: Hardware gain typically defaults to **1.0** as the pipeline-processed images already account for detector gain.
    - Python typically maintains DN units for internal consistency but supports these hardware factors for final flux calibration and validation.

### Components

1.  **`jwst1pass` (Python Package)**
    - **Location**: `jwst1pass_py/jwst1pass/`
    - **Diagnostics**: `jwst1pass_py/jwst1pass/diagnostics.py` — contains logic for star stamps, residual maps, and statistical summaries. Includes the `save_standard_plots` convenience function for organized output. **Note**: $\chi^2$ visualizations and residual ratios are mathematically corrected to account for the root-scaled `chi2_scale` attribute.
    - **CLI**: `jwst1pass_py/scripts/jwst1pass_run.py` — fully synchronized with the professional-grade `py1pass_run.py` (HST). It supports advanced overrides for sky annulus (`--sky_inner`, `--sky_outer`), noise parameters (`--gain`, `--read_noise`), and parallel processing (`--n_jobs`). Output resolution is robust, automatically deriving default paths based on the input image stem.
    - **Performance**: Numba-accelerated kernels in `core.py` (JIT compiled on first run).

2.  **Validation & Assessment Scripts**
    - **`scripts/assess_interp_accuracy.py`**: Quantifies the precision of PSF interpolation and finite-difference gradients.
    - **`tests/`**: Pytest-based unit test suite verifying the core solver, neighbor subtraction, and I/O integrity.

## Diagnostic & Validation Tools

The `jwst1pass_py` suite is logically synchronized with the professional-grade diagnostic tools in the improved HST implementation (`py1pass`):

-   **Logic Parity**: Shared algorithmic core with `py1pass` for residual stacking and stamp generation, including full support for `x_offset` and `y_offset` to handle dithered/multipass subtractions.
-   **Star Stamps**: Multi-panel plots (`_stamps.png`) showing raw data (neighbors removed), model, residuals, and pull ($\sigma$). 
    -   **Unified Masking**: DQ-flagged and sigma-clipped pixels are set to `NaN` and rendered against a **light gray background** (`#f0f0f0`). 
    -   **Boundary Visibility**: For NIRISS, **gray rectangles** at the edges of stamps represent **reference pixels** (typically the first/last 4 rows/cols), which are correctly excluded from the fit.
-   **PSF Residual Maps**: Spatially-stacked, `NaN`-safe residual maps (`_psf_resid.png`) revealing systematic mismatches across the detector grid.
- **Catalog Statistics**: Figures (`_stats.png`) tracking uncertainty vs. magnitude and $q$-fit. Stars are **color-coded by detector/chip** (via `_chip_ext`) to highlight per-detector systematic floors. A systematic floor overlay (from `estimate_systematic_floor`) shows the fitted floor lines on σ_x and σ_y panels.
- **Concentration Diagnostics**: `_concentration.png` from `plot_concentration_diagnostics` showing 1×1, 2×2, and 3×3 concentration vs magnitude with stellar locus, 68% band, and adaptive classification bounds.
- **PSF Perturbation**: `measure_psf_perturbation` + `plot_psf_perturbation` stack star residuals into an oversampled δP correction map with iterative refinement and Savitzky-Golay smoothing.
-   **Mathematical Consistency**: In diagnostic plots, the stored root-values ($\text{chi2}$ and $\text{chi2\_scale}$) are **squared for display** in the "Raw $\chi^2$ vs mag" panel. This ensures the plot correctly represents the variance-normalized distribution and that the correction curve accurately bisects the data. The "Applied chi²-scaling" plot displays the root-scale directly.
- **Output Organization**: 
 
    -   **Automatic Locality**: Outputs default to the input image's directory.
    -   **Flexible Plot Placement**: The `--plot_subfolder` option (default: `diagnostics`) manages plot organization. Set to `""` to save plots alongside the FITS files.
-   **Automated Summary**: `summarize_catalog` provides terminal diagnostics on discovery rates and crowding.


## Instrument & Filter Support

- **MIRI**: Full support for all broadband filters (F560W–F2550W).
- **NIRISS**: 
    - Supports all standard imaging filters.
    - **Filter Logic**: When the `FILTER` keyword is `CLEAR`, the routine automatically uses the `PUPIL` keyword to identify the active filter (e.g., F200W).

## JWST/MIRI Technical Mandates

Based on Libralato et al. (2024, PASP 136):

1.  **Empirical ePSFs**: Precision work MUST use the empirical ePSF models released with the 2024 study. Standard `WebbPSF` models currently lack critical detector effects like the **cruciform artifact** and interpixel capacitance.
2.  **Brighter-Fatter (BF) Effect**: Analysts MUST account for the BF effect in bright MIRI sources, which causes profiles to broaden and may lead to systematic residuals during subtraction.
3.  **Geometric Distortion (GD)**: MIRI photometry should use the 2024 GD corrections (polynomial + look-up table) which achieve ~1 mas precision.
4.  **Lyot Region**: ePSF models in the Lyot coronagraphic region are typically spatially constant due to lower star density; the imager region uses a 2x3 or 2x2 grid.

## Algorithmic Foundations (Anderson 2022)

- **One-Pass Philosophy**: The core logic prioritizes measuring isolated stars accurately in a single pass, avoiding the complexities of simultaneous multi-star fitting where possible.
- **Isolation Index (`h`)**: Detections are filtered by `HMIN`, the radius (in pixels) within which no brighter neighbor exists.
- **Quality of Fit (`q`)**: A primary metric for star-galaxy separation; $q \approx 0$ for stars, $q \gtrsim 0.1$ for poorly-fit or resolved objects.

## Validation & Cross-Implementation Results

The Python implementation (`jwst1pass_py`) has been rigorously validated against the reference Fortran code (`fortran_codes/jwst1pass-2024.02.01_v1d_LC.F`).

-   **Astrometric Precision**: Matches reference Fortran positions to within **< 0.001 pixels** (median residual ~0.0008 pix) for the finalized joint-solver implementation.
-   **Photometric Consistency**: Shows a perfect linear relationship with reference magnitudes. 
    -   **Note**: Python now operates in native MJy/sr (same as Fortran default `XFACTOR=1`), so no systematic magnitude offset from unit conversion should exist. Any residual offset reflects PSF normalization differences (e.g., NIRISS +0.65 mag from peak-normalized vs. sum-normalized ePSF models), not unit conversion.
-   **Empirical Calibration**: Official CRDS reference files (e.g., `jwst_niriss_gain_0006.fits`) provide pixel-level gain and noise maps. While the pipeline reference for NIRISS gain is ~1.61 e-/DN, the Fortran code uses **hardcoded empirical gains (~2.0 e-/DN)** for amplifier-specific corrections. This discrepancy (~20%) must be accounted for during end-to-end validation.
-   **Convergence, Sensitivity & Completeness**: 
    -   At high sensitivity (`FMIN=100` or lower), Python identifies and converges on significantly more stars in high-SNR regimes, while maintaining robust performance at low SNR.
    -   **Multi-pass Advantage**: The Python implementation consistently outperforms the one-pass Fortran reference in completeness. By utilizing **star-aware variance maps** and iterative neighbor subtraction, it successfully recovers sources in crowded regions that the Fortran version misses or rejects.
    -   **Benchmark (Draco NIRISS, Fmin=5)**: Confirmed coordinate parity with median residuals $\Delta x \approx 0.0023, \Delta y \approx 0.0024$ pixels.
-   **Diagnostic Visualization**:
    -   **Plot 6.9 (X-Residual Map)**: Spatial distribution map colored by `dx` (Python - Fortran) for good matches, highlighting sub-pixel systematic trends.
    -   **Mas-Scale Statistics**: RA and Dec residuals are now reported in milliarcseconds (mas) for high-precision validation, providing depth beyond pixel-level checks.
-   **Standardized Validation Columns**:
    -   **Python Outputs**: MUST include `xyuvmWsqf` (core photometry), raw `XY` (matching raw `x,y`), and explicit `x_gdc`, `y_gdc` (matching corrected `u,v`) to verify coordinate transformation integrity.
    -   **Fortran References**: SHOULD use `OUT=uvmqxyrdabgcXY` to provide a complete set of internal diagnostic and raw coordinate columns.

## GaiaJWST Pipeline Constraints & Legacy Parity

To ensure successful integration with the legacy GaiaHub pipeline and avoid systematic biases in absolute proper motion frames:

### 1. Coordinate Indexing & Legacy Parity
- **Python-to-Fortran Parity**: Python raw coordinates are 0-based. However, legacy Anderson-style catalogs (`.uvmqxyrd...` and `.XYmqxyrd`) and alignment tools (`xym2pm_GH.e`) REQUIRE **1-based indexing** for pixel coordinate columns.
- **GDC Parity**: The `apply_gdc` routine returns 1-based `u, v` coordinates because the underlying STDGDC lookup tables are defined on a 1-based pixel grid.
- **Catalog Parser**: The `make_XYmqxyrd` parser in `GaiaJWSTmod.py` uses fixed-width character slicing (e.g., `line[:100]`). Any modifications to the text catalog format MUST ensure that the first 8 columns occupy exactly 93 characters followed by 7 spaces for parity.

### 2. Alignment Stability (Sparse Fields)
- **Star Threshold**: For sparse fields like Draco dSph (~62 Gaia stars total), the default `min_stars_alignment` MUST be lowered to **20 stars** (down from the legacy 100-star default) to prevent transformation instability and image skipping.
- **Aggressive Clipping**: The `check_mat` logic applies a GMM-based outlier removal. To prevent a "vicious cycle" where iterative clipping reduces the star count below a solvable 6-parameter threshold, the routine now enforces a floor of `min_stars_alignment` during MAT file overwriting.

### 3. Statistical Completeness & Uncertainty
- **Single-Detection Stars**: Stars with only one detection ($N=1$) yield `NaN` standard deviations. They MUST fallback to measurement errors (`weighted_avg_error` for `mean`, and `weighted_avg_error * 1.25404` for `median`) to prevent them from being dropped from absolute reference frames.
- **Reduced Chi-Squared Scaling**: Final `wmean` uncertainties MUST be scaled by the square root of the reduced chi-squared ($\sqrt{\chi^2_{red}}$) when the external scatter between images exceeds the propagated internal errors. This ensures physically realistic error bars for the absolute PM frame.
- **Pandas Numerical Safety**: When performing arithmetic on Pandas Series (e.g., `scale_factor` derivation), use Pandas-native methods like `.clip(lower=1.0)` and `.pow(0.5)` to avoid `TypeError` with `numpy` universal functions in complex environments.

### 4. Proper Motion Reference Frames
- **Weighted Mean (wmean)**: Highly sensitive to alignment quality. Unstable image alignments (too few stars) can inject large systematic skews (e.g., -7 mas/yr).
- **Median Reliability**: The `median` frame is more robust against alignment outliers and is the preferred reference for dwarf galaxy studies where foreground contamination is high.

### 5. Data Loading Mandates
- **LNK/AMP Parsing**: Header parsing for legacy `.LNK` and `.AMP` files MUST use `split()` without arguments (e.g., `f.readline().split()`) to robustly handle arbitrary whitespace.
- **Zero Value Integrity**: Do NOT use `na_values=0.0` when reading quality catalogs (like `q_hst`). `0.0` is a valid "perfect fit" measurement and treating it as `NaN` will corrupt the statistical weighting weights.

## Foundational Mandates

1.  **Validation**: All changes to photometry or astrometry logic MUST be validated against the reference Fortran implementations in `fortran_codes/`. Comparison scripts (e.g., `Validation/cross_match_to_jwst1pass.py`) MUST account for the 0-based (Python) vs. 1-based (Fortran) coordinate difference and unit-driven magnitude offsets. Validation tables MUST include both raw and GDC-corrected coordinate sets for end-to-end verification.
2.  **Performance**: Python implementations (`jwst1pass`, `py1pass`) MUST utilize Numba-accelerated kernels for pixel-level operations.
3.  **Data Integrity**: Never modify raw data files. Outputs (catalogs, residuals) must be directed to external files.
4.  **Error Handling**:
    -   **Singular Systems**: Never crash on numerically singular normal equations. Catch `LinAlgError` and flag the star with `qfit=9.99` and inflated covariance.
    -   **Crowding**: Stars with `qfit > 0.5` are considered poor fits and MUST be skipped during PSF subtraction to avoid injecting artifacts into residuals.
    -   **Edge Sources**: Fitting routines MUST include shape checks for PSF evaluation windows to prevent broadcast errors for sources near detector boundaries.

5.  **PSF Consistency**: PSF evaluation MUST match the Anderson (2022) convention (WFC3 ISR 2022-05).
6.  **Covariance & Error Scale**: All photometry results MUST include the full 4x4 covariance matrix (flux, x, y, sky). The `StarRecord.chi2` attribute MUST store the **square root of the reduced chi-squared** to maintain mathematical parity with the improved HST suite.
- **Units**: Positions are in detector pixels. The engine uses 0-based indexing internally.
- **Legacy Indexing (1-based)**: To maintain parity with Jay Anderson's Fortran engine:
    - **Input**: `read_art_xym` automatically subtracts `1.0` from input `.xym` coordinates (assuming they are 1-based).
    - **Output**: All text-based catalogs (`.uvmq...` and `.XYmq...`) add `1.0` back to internal coordinates to produce 1-based output.
- **Robustness**: All data processing pipelines MUST handle `NaN` and `Inf` values in input science images. 

    - **Masking**: Data Quality (DQ) masking MUST be applied **before** identifying local maxima.
    - **Peak Shadowing**: `NaN` pixels must be pre-filtered to a very low value (e.g., `-1e10`) before running the maximum filter to prevent invalid pixels from suppressing nearby real peaks.
    - **Initial Guesses**: Background estimation must use `np.nanmedian` and explicit finite-value filtering to ensure the initial flux guess ($Peak - Sky$) never becomes `NaN`, which would otherwise crash the Newton-solver.
10. **Library Completeness**: The pipeline MUST skip processing of any FITS file if the corresponding PSF or GDC reference data is missing from the library. This prevents the generation of incomplete catalogs (e.g., zeroed-out GDC columns) that would fail downstream astrometric alignment.
11. **Type Safety**: Utility functions (like GDC applications) MUST preserve input dimensionality (e.g., returning scalars for scalar inputs) to prevent downstream NumPy deprecation errors.


## Advanced Photometric Calibration (`v2`)

The `v2` implementation introduces several architectural improvements to address the unique noise properties of JWST detectors and the undersampled nature of the PSF.

### 1. Advanced Noise Modeling
The `v2` implementation introduces a physically grounded noise model to address the unique properties of JWST detectors:
- **Physical Gain Scaling**: For NIRISS **and NIRCam** images in `MJy/sr`, the effective gain MUST be scaled by `xfactor = EFFEXPTM / PHOTMJSR` to convert pixel values to the electron scale required for Poisson statistics. The correct effective gain is `hardware_gain × xfactor` [e⁻/(MJy/sr)]. For NIRCam F115W this gives xfactor ≈ 311 and an effective gain of ~638, not the raw 2.05 e⁻/DN. Failure to scale causes a major weighting bias (bright-star pixels weighted ~1000× too low) and all-negative PSF residuals. The fix lives in `load_nircam_single` (io.py): compute xfactor from the FITS header and multiply `hardware_gain` before returning. NIRISS already handles this correctly in `load_image` (via the `elif units == 'MJY/SR'` branch at line ~397).
- **gain_map xfactor Scaling (NIRCam LW, June 2026)**: `load_nircam_single` scaled `effective_gain` by xfactor but left `gain_map` in raw e-/DN units. Because the Poisson variance term is `flux_MJysr * P / gain_map`, an unscaled gain_map (≈1.8 e-/DN instead of ≈553 e-/(MJy/sr)) inflated the Poisson term by ~306× for every pixel — making PSF wings as heavily weighted as the core, giving deflated chi² (≈0.8 instead of ≈5), and hiding real PSF residuals. **Fix (io.py, `load_nircam_single`):** `if bunit == 'MJY/SR' and gain_map is not None: gain_map = gain_map * xfactor`. Both `effective_gain` and `gain_map` must be scaled together — applying xfactor to only one produces an inconsistent noise model.
- **Spatial Gain Map**: Instead of a global average, it uses a 2D map of amplifier-specific gains. This ensures the Poisson noise term ($\text{Flux}/G_{eff}$) is correctly weighted across amplifier boundaries, eliminating discrete statistic clustering.
- **NaN-Safe Solver**: The iterative fitting solver includes `np.isfinite` guards within the sigma-clipping loop to prevent coordinate drift crashes on edge-of-detector stars or extreme noise.
- **1/f Row-Noise Estimation**: Iteratively estimates horizontal banding (1/f noise) from residuals and incorporates it into the variance model. This "down-weights" noisy rows, improving faint star detection.
- **Hybrid Variance**: When the pipeline `ERR` map is available, variance = `ERR² + (0.01·flux·P)²`. Self-Poisson is NOT re-added (ERR already contains it). Between passes, `build_variance_image` adds neighbor-star Poisson on top of the base ERR² map.

### 2. Core-Window QFit Alignment
The `qfit` (Quality of Fit) calculation has been aligned with the reference Fortran logic to ensure consistent star/galaxy separation:
- **Central 5x5 Window**: Instead of the full fit window (e.g., 11x11), `qfit` is calculated using only the central 5x5 pixels.
- **Local Flux Re-optimization**: A local flux scaling factor ($zu$) is calculated specifically for this 5x5 core. This ensures the metric is dominated by the star's shape rather than global flux errors or background noise in the wings.
- **Impact**: This avoids "noise dilution," resulting in `qfit` values that are physically consistent with the original Fortran implementation (e.g., ~57% of bright stars achieving `qfit < 0.1`).

### 3. Advanced Visualization
The `v2` diagnostic suite enhances the interpretability of solver performance:
- **Per-chip / per-population coloring**: In `_stats.png`, σ_x and σ_y scatter panels colour points by detector chip (`_chip_ext`) to reveal per-chip systematic floors. The qfit panel colours star candidates by qfit (plasma colormap); non-star candidates are shown as grey ×. The concentration panel (2×2) and qfit-vs-concentration panel use star/non-star colouring. **Note:** Earlier drafts of this CLAUDE.md specified n_iter/viridis coloring; that was replaced during the pypass diagnostic port (July 2026) with chip/star coloring, which is more informative for astrometric floor analysis.
- **Systematic floor overlay**: `estimate_systematic_floor` (in `diagnostics.py`) fits the model `σ = sqrt(floor² + (A·10^(0.2m))² + (C·10^(0.4m))²)` to magnitude-binned σ_x and σ_y. The fitted floor lines (Option A nonlinear fit, Option B empirical bright-end median) are overlaid on the σ panels in `_stats.png`.
- **Concentration diagnostics**: `plot_concentration_diagnostics` produces a separate `_concentration.png` with 1×1, 2×2, and 3×3 concentration vs magnitude panels, each showing the stellar locus (binned median ± 68%) and adaptive classification bounds.
- **PSF perturbation**: `measure_psf_perturbation` drizzles star fit residuals onto the oversampled PSF grid (iterative, σ-clipped, Savitzky-Golay smoothed) to measure δP — a spatially uniform correction to the PSF model. `plot_psf_perturbation` shows 3D surface plots of P, δP, P+δP, and δP/P, plus coverage map and cross-sections. Both functions live in `diagnostics.py`.
- **Population Reliability**: The spatial residual map (`_psf_resid.png`) uses `is_star_candidate` filtering and the `mask` parameter (JWST DQ bit-0) to exclude bad pixels before weighted accumulation.
- **Discrete q-band Interpretation**: When an image has very few detector reads (e.g., 3 groups, ~150 s), noise is read-noise dominated and nearly uniform across the detector. Stars in the same iteration group share near-identical covariances, producing discrete horizontal bands in the σ_x and q diagnostic panels. This is physical, not a bug. Long exposures (many groups, Poisson-dominated) show continuous scatter instead.

### 4. CRDS Reference File Auto-Download (Added June 2026)

`jwst1pass/io.py` now automatically fetches missing gain and read-noise reference files from STScI CRDS.

**How it works:**
- `_cal.fits` headers contain `R_GAIN: crds://jwst_nircam_gain_0096.fits` and `R_READNO: crds://jwst_nircam_readnoise_0248.fits`.
- `_get_gain_map` and `_get_read_noise_map` check `lib/R_GAIN/<name>` and `lib/R_READNO/<name>`. If the file is absent, they call `_fetch_crds_reference(ref_name, target_dir)` which downloads from `https://jwst-crds.stsci.edu/unchecked_get/references/jwst/<ref_name>` using `urllib.request`.
- A `.downloading` temp-file suffix prevents partial downloads from being used.
- On failure the download warning is printed and the function falls back to the hardcoded scalar or `None` (same as before).

**Path-case fix:** The old code used `lib_dir/R_Gain/` (mixed case) which never matched the actual `lib/R_GAIN/` directory. Both functions now use `R_GAIN` and `R_READNO` (all-caps) to match the real directory names.

**NIRCam spatial maps now enabled:** `load_nircam_single` was previously hardcoded to return `(None, None)` for `gain_map` and `read_noise_map`. It now calls `_get_gain_map` and `_get_read_noise_map` with `lib_dir`, so NIRCam images benefit from 2D spatial noise maps when the reference files are available (either already downloaded or auto-fetched). The `load_image` → `load_nircam_single` call passes `lib_dir` through.

**DN→electrons conversion:** `_get_read_noise_map` applies the DN→electrons conversion when `BUNIT == 'DN'`, or when `BUNIT` is blank and the instrument is MIRI (MIRI files such as `0085` store DN values but omit the BUNIT keyword). Conversion uses the spatial gain map when available, or per-instrument scalar fallbacks (NIRCam: 2.05 e-/DN, NIRISS: 1.9835 e-/DN, MIRI: 4.5 e-/DN).

**Reference file availability (confirmed June 2026):**

| Instrument | Gain file | In `lib/R_GAIN`? | RN file | In `lib/R_READNO`? |
|---|---|---|---|---|
| NIRISS | `jwst_niriss_gain_0006.fits` | ✓ | `jwst_niriss_readnoise_0005.fits` | ✓ |
| MIRI | `jwst_miri_gain_0040.fits` | ✓ | `jwst_miri_readnoise_0085.fits` | ✓ |
| NIRCam SW (e.g. NRCA4) | `jwst_nircam_gain_0096.fits` | ✓ | `jwst_nircam_readnoise_0248.fits` | Missing → auto-download |
| NIRCam LW (NRCALONG) | `jwst_nircam_gain_0097.fits` | Missing → auto-download | `jwst_nircam_readnoise_0266.fits` | Missing → auto-download |

All gain files have `BUNIT=ELECTRONS/DN` and are returned as-is. All RN files are in DN and are converted to electrons.

### 5. Dead-Pixel Mask in PSF Residual Map (Fix, June 2026)

**Bug:** `plot_psf_residual_map` in `diagnostics.py` and `diagnostics_Q.py` previously produced all-negative residual maps (global mean ≈ −26%, RMS ≈ 36%) for NIRCam MJy/sr data.

**Root cause:** `load_nircam_single` stamps dead pixels with `LOFLAG−1 = −101` (a finite value, not NaN). The variance map (`var_map`) is built from `ERR²`, which does **not** propagate the LOFLAG sentinel — so dead pixels retain a normal, small variance and receive full weight in the weighted average. For any star whose 11×11 fitting window overlaps a dead pixel, the term `r_norm = (−101 − sky) / (flux × psf_peak)` is enormously negative. On this image, 33% of good stars (931/2776) had at least one dead pixel in their window.

**Fix:** Added a `mask` parameter to both `plot_psf_residual_map` functions. Inside the per-star loop, `bad |= mask[y_lo:y_hi, x_lo:x_hi].astype(bool)` is applied after the existing `clipped_mask` check. `jwst1pass_run.py` now passes `mask=_diag_mask` (the bad-pixel mask returned by `load_nircam_single`) to both calls.

**Result:** Global mean +0.20%, RMS 0.55%, fraction negative 21% — balanced, unbiased residuals confirming correct photometry.

**Mandate:** Any future call to `plot_psf_residual_map` that uses a `noise_map` from `var_maps` MUST also pass the corresponding pixel mask. Without it, dead/LOFLAG pixels contaminate the weighted average with extreme negative values that drown out the true PSF residual signal.

### 6. NIRCam LW Detector Prefix Fix (June 2026)

**Bug:** NIRCam long-wavelength exposures (NRCALONG, NRCBLONG) produced no outputs at all — the exposure directory was left with only the input `_cal.fits`.

**Root cause:** `_DETECTOR_PREFIX` in `io.py` mapped `('NIRCAM', 'NRCALONG') → 'NRCALONG'` and `('NIRCAM', 'NRCBLONG') → 'NRCBLONG'`. Both `find_psf` and `find_gdc` used this prefix to construct filenames, e.g. `STDPSF_NRCALONG_F277W.fits`. The actual library files use the abbreviated 5-character prefix: `STDPSF_NRCAL_F277W.fits` and `STDPSF_NRCBL_F277W.fits`. Every lookup raised `FileNotFoundError`, which propagated through the subprocess call in `GaiaJWSTmod.py` silently (no return-code check), so the failure was invisible.

**Fix** (`io.py`, `_DETECTOR_PREFIX`):
```python
('NIRCAM', 'NRCALONG'): 'NRCAL',   # was 'NRCALONG'
('NIRCAM', 'NRCBLONG'): 'NRCBL',   # was 'NRCBLONG'
```

**Scope:** Affects both PSF lookup (`lib/STDPSFs/NIRCam/LWC/STDPSF_NRCAL_*.fits`) and GDC lookup (`lib/STDGDCs/NIRCam/LWC/STDGDC_NRCAL_*.fits`). NIRCam SW detectors (NRCA1–4, NRCB1–4) were unaffected — their prefixes already matched the library filenames.

**Mandate:** The `_DETECTOR_PREFIX` map MUST use the 5-character abbreviated form (`NRCAL`, `NRCBL`) for LW detectors, matching the STScI STDPSF/STDGDC naming convention.

### 7. DQ Extension Fix for NIRISS and MIRI (June 2026)

**Bug:** For NIRISS and MIRI, the DQ mask passed to `find_sources` was effectively empty — virtually no pixels were masked before the local-maxima search for source candidates.

**Root cause:** `get_chip_config` returned `dq_ext=2` for all non-NIRCam instruments. In standard JWST `_cal.fits` files the extension layout is:

| Index | Name | dtype |
|---|---|---|
| 1 | SCI | float32 |
| 2 | ERR | float32 |
| 3 | DQ | int32 |

Extension 2 is ERR, not DQ. `_load_dq_mask` received float32 ERR data, converted it to int32, and returned `dq != 0`. Because typical MJy/sr ERR values are <<1.0, truncation to int32 produces zero for nearly every pixel, giving a near-empty mask. The real DQ extension (index 3, with genuine hot-pixel, cosmic-ray, and DO_NOT_USE flags) was never read.

**Observed impact:**
- NIRISS: old code masked 19 pixels (0.00%); correct DQ masks 159,768 pixels (3.81%)
- MIRI: old code masked 565 pixels (0.05%); correct DQ masks 345,638 pixels (32.71%), dominated by bit 0 (DO_NOT_USE) — the known high dead-pixel fraction of the MIRI detector

Without proper DQ masking, hot pixels and dead pixels competed with real stars as local-maxima candidates in `find_sources`, corrupting the source list before any PSF fitting began.

**Fix** (`io.py`, `get_chip_config`):
```python
return [(1, None, 0.0)]  # dq_ext=None → load_image finds DQ by name (ext 3)
```

With `dq_ext=None`, the existing auto-detection in `load_image` scans extensions by name (`'DQ' in hdul[i].name.upper()`), correctly resolving extension 3 for both NIRISS and MIRI. NIRCam was already correct: `load_image` routes 2048×2048 NIRCam images to `load_nircam_single`, which finds the DQ extension by name independently.

**Mandate:** `get_chip_config` MUST NOT hardcode a numeric `dq_ext` index. The DQ extension MUST be found by name so the code is robust to any future change in JWST pipeline output layout.

### 8. qfit Convention and Masked-Pixel Handling (updated July 2026)

**qfit is a genuine quality metric** — `small ≈ good fit`, `large ≈ poor fit`, consistent with pypass (BP3M's HST engine). The previous convention of forcing `qfit = 0.0` whenever `n_sat > 0` (any DQ-flagged pixel in the fit window) has been **removed**.

#### Why the sentinel was removed
JWST CAL files have many DQ-flagged pixels (cosmic rays, hot/dead pixels). In a typical NIRISS exposure, 84% of detected sources have at least one masked pixel in their 11×11 fit window. Forcing `qfit = 0.0` for all of them made the 25th-percentile of qfit equal 0.0, collapsing the adaptive qfit threshold in `classify_stars` to 0.0 and rejecting every source — producing 0 stellar candidates.

#### New behaviour
- **`qfit`** is always computed from PSF residuals over the good (unmasked) pixels in the fit window. A source with one flagged pixel and clean residuals on the remaining ~120 pixels correctly receives a low qfit.
- **`n_sat`** (count of DQ-flagged pixels in the fit window) is still computed and stored in `StarRecord.n_sat`. It is available for downstream filtering but does NOT override qfit.
- **Real magnitude**: `mag_from_flux` is always called. No `mag = 99.99` sentinel is used.

#### Impact on `xym2pm_GH.e` (Fortran alignment engine)
The Fortran engine (`xym2pm_GH.F`) uses `qfit = 0.0` as the "international sign of saturation" (line 17083: `qr = 0 ! international sign of saturation`). Stars with `qfit = 0.0` in the `.XYmqxyrd` input are kept but flagged as saturated for special handling in the alignment. **With the new convention:**
- Stars with genuinely good fits now have `qfit ≈ 0.01–0.1` instead of `0.0` — `xym2pm_GH` will no longer treat them as saturated, which is correct.
- Truly saturated stars (bright cores overflowing into neighbouring pixels) will have elevated qfit from the PSF mismatch in their wings, and will naturally fail quality cuts in the Fortran engine (`qfit > threshold`).
- **If you run `jwst1pass_py_v2` output through `xym2pm_GH.e`, verify that the saturation handling is acceptable.** The Fortran code should be unaffected for faint-to-medium stars; only the brightest saturated sources change category from "saturated (qfit=0)" to "poor fit (qfit=large)".

**Mandate:** Do NOT restore the `qfit = 0.0` sentinel for `n_sat > 0`. Use `n_sat` directly if downstream tools need a boolean saturated flag.

### 9. Stars with Masked Pixels in Diagnostic Plots (updated July 2026)

Sources with DQ-flagged pixels (`n_sat > 0`) are **included** in all diagnostic plot functions. No `n_sat == 0` filter is applied.

With the new qfit convention (§8), sources with `n_sat > 0` now have genuine qfit values:
- Good fits with a few masked pixels: low qfit → appear in `plot_psf_residual_map` (`qfit < 0.1`) and stats plots (`q < 2`).
- Truly saturated bright stars: elevated qfit from PSF-wing mismatch → naturally excluded from the PSF residual map but visible in stats plots.

**Mandate:** Do NOT add `n_sat == 0` exclusion guards to diagnostic plot filters.

## JWST1Pass Python v2 (Pure Translation Edition)

### Core Mission
This package is a **direct numerical translation** of Jay Anderson's `jwst1pass.F` Fortran engine. Its primary goal is to produce catalogs that are bit-for-bit (or within floating-point tolerance) compatible with the legacy Fortran pipeline used in GaiaJWST.

### Architectural Mandates (The "Pure" Rules)

### 1. Magnitude Scale & Units
- **Convention**: Magnitudes follow the legacy **"Total Flux" convention**. No division by `EFFEXPTM` is performed during magnitude calculation, matching the numerical results of the Fortran `jwst1pass` engine.
- **Instrument-Specific Scaling**:
    - **MIRI**: Stays in native **MJy/sr** units, matching the Fortran default (`XFACTOR=1`). Effective gain is scaled: `g_eff = hardware_gain × xfactor`.
    - **NIRISS**: Stays in native **MJy/sr** units. Same treatment as MIRI.
    - **NIRCam**: Stays in native **MJy/sr** units. Same treatment.
- **Reporting Parity**: `flux_unit_conversion = 1.0` for all instruments. Flux from the fitter is already in MJy/sr and is passed directly to `mag_from_flux`. `exposure_time = 1.0` for MIRI and NIRISS (rate-unit images require no exposure-time division in the magnitude formula).
- **Instrumental Offsets**:
    - **MIRI**: Achieves **~0.00 median offset** against Fortran because the 2024 empirical models (Libralato) follow the modern **"Sum=1.0" normalization**.
    - **NIRISS**: Exhibits a stable **+0.65 magnitude shift** against Fortran. This is an instrumental constant resulting from the difference between the legacy **"Peak-Normalized" models** (Jay Anderson 2022) and the modern **"Integral-Normalized" solver** (Sum=1.0) used in Python.
- **Flux Guards**: Implement Fortran-exact guards: floor flux at `0.01` and cap extreme values at `1.00` if $> 10^{19}$.

### 2. Artificial Star (ART) Injection
- **Flux Recovery**: Images are in MJy/sr (rate units), so `exposure_time = 1.0` for MIRI and NIRISS. Injected flux is `flux_MJysr = 1.0 * 10**((ZP - m) / 2.5)`, matching the Fortran `fr = 10**(-mr/2.5)` intent (no EFFEXPTM division).
- **Noise Model**: Noise is injected as Poisson noise in MJy/sr: $\sigma = \sqrt{\text{flux} / g_\text{eff}}$ where `g_eff = hardware_gain × xfactor [e⁻/(MJy/sr)]`.

### 3. Solver Dynamics & Convergence
- **Numerical Solver (Newton-Raphson)**: Python uses a 4-parameter joint Newton-Raphson solver. 
- **Stability Breakthrough (99% Convergence)**: To achieve ~99% convergence in sparse, noisy fields (e.g. Draco dSph), the solver uses **Post-Convergence Sigma Clipping**. 
    - The Newton-loop first solves for $(x, y, f, s)$ using all pixels. 
    - Once converged, it performs 2 refinement iterations to reject outliers (cosmic rays, bad pixels) and re-solves. 
    - This separation prevents the "oscillating Newton step" common in noisy JWST data.
- **Safety Clamps**:
    - **Positional Bailout**: If a Newton step exceeds **2 * half-width** pixels, the star is marked as diverging.
    - **Relative Flux Floor**: Prevent flux from crashing more than **75%** in a single iteration.
- **Pre-Masking Mandate**: Data Quality (DQ) masking MUST be applied **before** identifying local maxima. Setting invalid pixels to `-inf` ensures they cannot be local maxima and cannot suppress nearby real peaks.

### 4. Catalog Structure & BP3M Integration
- **Covariance Serialization**: Columns 9, 10, and 11 of the text catalogs (`.uvmqxyrdabgcenW`) are populated with raw fit uncertainties: **$\sigma_x$**, **$\sigma_y$**, and **$\rho_{xy}$** (correlation). 
- **BP3M Compatibility**: This serialization is required to avoid `NaN` crashes in the `XY2PM` and `BP3M` engines and to allow the derivation of valid `med_mult` (median error multiplier) values (~0.8-1.0).
- **Converged vs Failed**: Always output two catalogs:
  1. `..._catalog.fits` (Converged detections only)
  2. `..._failed_catalog.fits` (Detections that hit `max_iter` or other failures)
- **Legacy Text Formats**: Catalogs MUST use **Fortran 1-based indexing** for coordinates.
- **Fixed-Width Parsing**: The text catalog first 8 columns MUST occupy exactly 93 characters followed by 7 spaces for parity with `GaiaJWSTmod.py`.

### Validation Standards
- **Astrometric Precision**: Python vs Fortran residuals must be **< 0.001 pixels** (median).
- **Photometric Agreement**: Empirical mag offset must be constant across all star populations (Unified Line).
- **Tooling**: Use `Validation/cross_match_to_jwst1pass.py` to verify every major engine change.

#### 4. Quality Metrics (q vs Q)
- **`q` (5x5)**: Small-window quality metric. Small ≈ good fit; large ≈ poor fit or extended source. Consistent with pypass convention.
- **`Q` (Full)**: Large-window quality metric for diagnostic purposes.
- **Masked-pixel sources**: Sources with DQ-flagged pixels (`n_sat > 0`) receive genuine qfit from the good pixels — no sentinel override. See §8.

### 10. Performance & Robustness Improvements (June 2026)

#### enforce_hmin O(N log N) Replacement (`multipass.py`)

**Previous implementation:** O(N²) — iterated over every (i, j) pair to check whether star j was within `hmin` pixels of the brighter star i. For 5,000+ stars this inner loop dominated runtime.

**New implementation:** Build one `cKDTree` from all candidate positions (sorted bright-to-faint). For each surviving star i, call `query_ball_point(coords[i], hmin)` to mark all fainter stars within hmin as rejected. The tree is built once; the per-star cost is O(log N) on average.

**Result parity:** Both algorithms implement the same greedy bright-first suppression — the `j > i` guard (only reject fainter, i.e. higher-index entries) ensures identical output for any input list.

#### PSF Cache Fully Wired (`utils.py`, `multipass.py`)

**Previous state:** `subtract_stars` and `build_variance_image` accepted a `psf_cache=None` parameter and passed it to `_psf_window`, but `interpolate_psf` (the actual interpolation function) had no `_cache` parameter — so the dict was silently ignored and every call recomputed the Catmull-Rom bicubic from scratch.

**Fix:** Added `_cache` parameter to `interpolate_psf` in `utils.py`. Key is `(int(x_det)//5, int(y_det)//5)`; max 2048 entries. `run_photometry` in `multipass.py` now creates `psf_cache: dict = {}` at the top of each call and passes it through to all four `subtract_stars`/`build_variance_image` invocations. PSF variation across a 5-pixel cell is negligible for photometry.

#### Dead `history` List Removed (`core.py`)

A `history = []` list in the Newton-loop of `fit_star` was allocated and appended to each iteration but only used in a commented-out debug print. Removed the declaration and all `history.append(...)` calls to eliminate the per-star allocation overhead.

### Validation Standards
- **Astrometric Precision**: Python vs Fortran residuals must be **< 0.01 pixels** (median).
- **Photometric Agreement**: Empirical mag offset must be **< 0.05 mag** (with $ZP=0$ on both sides).
- **Tooling**: Use `Validation/cross_match_to_jwst1pass.py` to verify every major engine change.

### 11. Detection Threshold and AB Magnitude (June 2026, updated July 2026)

#### Two-parameter detection threshold (`fmin_thresh` / `mag_limit`)

`run_photometry_fits` now accepts three threshold parameters instead of a single raw `fmin`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fmin` | `None` | Direct MJy/sr override — bypasses the two-param logic entirely (legacy compatibility) |
| `fmin_thresh` | `5.0` | Hard lower floor in MJy/sr; sets the minimum detectable flux regardless of `mag_limit` |
| `mag_limit` | `None` | Faint AB magnitude limit; converted to MJy/sr using ZP_AB derived from `PIXAR_SR` |

Effective fmin resolution (same logic in both `run_photometry_fits` and `_run_nircam_meta`):
```
_zp_for_mag = ZP_AB (from PIXAR_SR) if available, else zero_point
if fmin is not None:        → use fmin directly (legacy override)
elif mag_limit is not None: → fmin_eff = max(10^((_zp_for_mag - mag_limit) / 2.5), fmin_thresh)
else:                       → fmin_eff = fmin_thresh
```

**`mag_limit` is interpreted in AB magnitudes** (via `ZP_AB = -2.5 * log10(PIXAR_SR × 1e6 / 3631)`), not in the instrumental system of `zero_point`. `zero_point` remains 0.0 for Fortran-compatible instrumental `mag` column output and must not be used for the threshold conversion. `ZP_AB` is computed from the FITS header before the fmin block in both `run_photometry_fits` and `_run_nircam_meta`, so it is available for the conversion. If `PIXAR_SR` is absent, falls back to `zero_point`.

`fmin_thresh` is a fixed flux floor in MJy/sr — it does **not** adapt to the local noise level. In dense/crowded fields where the effective per-pixel scatter is elevated by overlapping PSF wings, `fmin_thresh=5` may correspond to only ~3–4σ locally, pulling in marginal detections. See §sky_std below.

Per-chip computation is NOT needed for JWST because `get_chip_config` always returns a single chip and `ZP_AB` is derived from the single `PIXAR_SR` header keyword. `_run_nircam_meta` computes `_fmin_effective` once before the 8-chip loop.

CLI flags: `--fmin`, `--fmin_thresh`, `--mag_limit` in `scripts/jwst1pass_run.py`.

#### Field density pre-check and sky_std

Before running the full PSF fit, field density can be assessed in seconds by calling `find_sources` alone (no fitting). Two useful metrics:

- **Global sky_std** (`np.nanstd` of all unmasked pixels): a coarse image-level proxy. In a sparse field it reflects readout + sky Poisson noise; in a dense field it is inflated by overlapping PSF wings. Useful as a quick density flag before committing to a full fit.
- **Per-source sky sigma** (`sigs` returned by `find_sources` via `estimate_sky_batch`): sigma-clipped std of pixels in the sky annulus (radii `sky_inner`–`sky_outer`) around each candidate. This is the noise the pipeline actually uses per source — but `fmin_thresh` is a raw flux floor, not a sigma multiple, so the pipeline does not divide by `sigs` at the detection stage.

Observed values for NIRCam F200W 150s exposures:

| Field type | global sky_std | candidates at fmin=5 | quality |
|---|---|---|---|
| Sparse (NRCA4) | ~0.54 MJy/sr | ~26 | q_med≈0.05, chi²_med≈1.1 |
| Dense (NRCA3/B3) | ~1.2–1.4 MJy/sr | ~1300–1370 | q_med≈1.1, chi²_med≈3–4 |

The elevated q/chi² in dense fields is **not** a PSF model problem — bright stars (AB < 24) in those same fields fit with q_med≈0.025, identical to the sparse field. The poor metrics come entirely from faint sources near the noise floor that are marginally detected noise peaks or unresolved blends.

#### AB magnitude zero point from `PIXAR_SR`

The output catalog now contains a `mag_ab` column alongside the Fortran-compatible instrumental `mag` column:

- `mag` — instrumental: `zero_point - 2.5 * log10(flux_MJy/sr)`. Unchanged; Fortran-compatible with `zero_point=0.0`.
- `mag_ab` — calibrated AB: `-2.5 * log10(flux_MJy/sr) + ZP_AB`, where `ZP_AB = -2.5 * log10(PIXAR_SR × 1e6 / 3631)`.

`PIXAR_SR` (pixel solid angle in sr) is read from the SCI extension header (with primary header fallback). If absent, `mag_ab` is `NaN`. `ZP_AB` is stored in the table metadata as `tab.meta['ZP_AB']`.

Derivation: the PSF-fitted flux F [MJy/sr] × PIXAR_SR [sr] × 10^6 [Jy/MJy] gives total source flux in Jy. Substituting into `m_AB = -2.5 * log10(F_Jy / 3631)` and separating terms yields `ZP_AB`. This assumes the STDPSF is normalised to unit integral; `psf_frac` in the output catalog provides the aperture-correction factor for users who need it.

Note: `mag_ab` is NOT corrected for GDC pixel-area variation (`mc`). `PIXAR_SR` is a single average solid angle from the header, already an approximation of the same effect.

### 12. Concentration Metrics and Star/Galaxy Classification (July 2026)

`jwst1pass_py_v2` now computes concentration metrics and classifies every detected source as a likely star or non-star, matching the `pypass` (BP3M HST engine) interface. This is required for BP3M cross-matching: `gaia_cross_match/cross_match.py` hard-skips any image whose catalog lacks `is_star_candidate`.

#### New `StarRecord` fields

```python
concentration      # 1×1: (data_center - sky) / (flux × PSF_peak)
concentration_2x2  # 2×2: sum(data - sky) / (flux × sum(PSF)) over 4 pixels bracketing sub-pixel centre
concentration_3x3  # 3×3: sum(data - sky) / (flux × sum(PSF)) over 3×3 pixels centred at rounded position
n_conc_1x1         # number of unmasked pixels entering concentration (0 or 1)
n_conc_2x2         # number of unmasked pixels entering concentration_2x2 (0–4)
n_conc_3x3         # number of unmasked pixels entering concentration_3x3 (0–9)
is_star_candidate  # bool — set by classify_stars() after all passes complete
```

Interpretation: stars ≈ 1.0; extended sources (galaxies) < 1 (flux spread over more pixels than the PSF predicts); cosmic rays / noise spikes > 1 (flux concentrated in fewer pixels than the PSF predicts). Only unmasked (DQ-valid) pixels enter both numerator and denominator.

#### Where concentration is computed (`core.py`, `fit_star`)

After the Newton loop converges and σ-clipping finishes, a block computes all three metrics using `_eval_psf_grad_fast` — the same PSF evaluation function already used in the fitter. The 2×2 aperture brackets the fitted sub-pixel position; the 3×3 aperture is centred at the rounded pixel position. Pixels outside the image or masked by DQ are excluded; if too few good pixels remain (< 2 for 2×2, < 5 for 3×3), the metric is set to `NaN` (treated as "pass" by the classifier).

#### Classification (`core.py`, after `compute_neighbor_stats`)

`_conc_adaptive_bounds()` and `classify_stars()` are **verbatim copies** of the identically-named functions in `bp3m/pypass/core.py` (lines 1275–1444). Algorithm:

1. **Seed** on 2×2 concentration with fixed window `[conc_lo, 1/conc_lo]` + global `qfit < 1.5` cap + magnitude-adaptive qfit threshold (25th-percentile × 4 within 0.5-mag bins)
2. **Iterate** (up to 10 rounds): compute per-magnitude adaptive bounds from the current stellar locus (median ± 4 × half-68%-spread, floored at 0.01) for all three metrics; re-classify; repeat until stable

#### Parameter threading

`conc_limit` (default 0.9) flows through:
```
run_photometry_fits(conc_limit=0.9)
  → _run_nircam_meta(..., conc_limit=conc_limit)   # NIRCam 8-chip path
  → run_photometry(..., conc_limit=conc_limit)       # multipass.py
      → classify_stars(all_records, conc_lo=conc_limit)
```

#### Validation (Draco NIRISS F200W, July 2026)

| Metric | Value |
|--------|-------|
| Total detections | 515 |
| Star candidates (`is_star_candidate=True`) | 81 (15.7%) |
| concentration_2x2 median (all sources) | 6.8 — dominated by noise spikes near detection limit |
| concentration_2x2 range for candidates | ~0.8–1.3 — genuine stellar locus |
| Sources with `n_sat > 0` (masked pixels in window) | 434 (84%) — correctly handled by genuine qfit |

The high overall concentration median is expected: at `fmin_thresh=5.0 MJy/sr` most detections near the noise floor are single-pixel noise spikes with concentration >> 1. The 81 stellar candidates all sit in the locus near concentration = 1.

#### BP3M integration (completed July 2026)

`jwst1pass_py_v2` is now fully wired into BP3M via `bp3m/bp3m/pipeline/psf_fitting_cal.py`, which is a JWST-only module (HST uses `psf_fitting.py`). `run_psf_fitting` calls `_fit_one_image` (a thin wrapper) → `_fit_one_image_jwst`, which:

1. Reads `PIXAR_SR` from the FITS primary header and computes `zero_point = -2.5 * log10(PIXAR_SR × 1e6 / 3631)` (AB zero point, per-image). Falls back to `params.get('zero_point', 0.0)` if `PIXAR_SR` is absent.
2. Calls `jwst1pass_py_v2.jwst1pass.io.run_photometry_fits` (with `conc_limit`, `hw`, `mag_limit=28.0`, `zero_point`, etc.)
3. Calls `_build_jwst_catalog_table(records, zero_point, ...)` — the column adapter — which:
   - Renames `q → qfit` and adds `mag_st_gdc = mag_gdc` (cross_match.py hard requirements)
   - Adds `is_star_candidate`, `concentration*`, `n_conc_*`, `chip_ext` from record attributes
   - Adds `eps_psf = NaN`, `sigma_*_model = NaN` (not computed by jwst1pass)
   - Propagates `sigma_floor_x/y`, `eps_flux`, `floor_params` from `estimate_systematic_floor`
4. Writes catalog, params sidecar, residual FITS (DQ + sigma-clip masks), and all diagnostic plots matching the HST pipeline output set

`_JWST_DEFAULTS` in `psf_fitting_cal.py` provides JWST-specific defaults: `fmin_thresh=5.0`, `hmin=5`, `half_width=5`, `mag_limit=28.0`, `n_passes=2`, `sky_inner=4`, `sky_outer=8`, `conc_limit=0.9`.

`_ensure_jwst1pass()` in `psf_fitting_cal.py` searches for jwst1pass_py_v2 at `$JWST1PASS_DIR` then at `../../GaiaWebb-master/jwst1pass_py_v2` relative to the bp3m package root. Set `JWST1PASS_DIR` to override.

**Mandate:** Do not use `PHOTMJSR` for the ZP formula. `PHOTMJSR` is the DN/s → MJy/sr conversion that was already applied by the JWST pipeline to produce the `_cal.fits` data — it is not a photometric calibration constant for converting MJy/sr to magnitudes.
