# Project Overview: GaiaWebb Python Pipeline

This directory contains the high-level Python coordination scripts for the GaiaWebb pipeline, which integrates JWST/HST observations with Gaia astrometric data to derive absolute proper motions.

## Key Components

### 1. `GaiaJWST.py`
- **Purpose**: The main entry point for the JWST-Gaia alignment pipeline.
- **Workflow**: 
    1. Downloads/locates JWST observations.
    2. Queries Gaia for reference stars.
    3. Triggers source detection (via `jwst1pass`).
    4. Orchestrates iterative astrometric alignment (via `xym2pm_GH`).
    5. Computes final error-weighted average proper motions.
    6. Generates scientific diagnostic plots.

### 2. `GaiaJWSTmod.py`
- **Purpose**: The core functional library for `GaiaJWST.py`.
- **Maintenance History (May 2026)**:
    - **WCS Alignment Fix**: Refactored `xym2pm_GH` coordination logic to use detector-specific WCS coordinates (`CRVAL1/2`) and reference pixels (`CRPIX1/2`) from image extensions instead of telescope boresight (`TARG_RA/DEC`). This reduced initial coordinate offsets from >2,000 pixels to <3 pixels, enabling robust star matching in JWST fields.
    - **Numerical Stability**: Fixed multiple `UnboundLocalError` bugs where variables like `match` or `mean` were referenced before initialization, particularly during alignment failures or plotting.
    - **TAP Mirror Fallback**: Implemented robust Gaia Archive querying with automatic fallback to the Heidelberg mirror and support for synchronous `launch_job` execution when asynchronous services are offline.
    - **Gaia Query Retry Logic (June 2026)**: `gaia_query` now retries transient network failures (`ConnectionResetError`, etc.) up to 3 times per mirror, with linear backoff (10 s, 20 s, 30 s), before falling over to the Heidelberg mirror. Both mirrors now use `launch_job_async` to avoid the 2000-row cap that `launch_job` (sync) imposes. This was triggered by dense fields (e.g., Terzan 5, ~147k Gaia stars in 0.5°) causing the ESA TAP server to drop SSL connections during job-phase polling, silently preventing the Gaia CSV from being written.
    - **MAST Instrument Coverage Fix (June 2026)**: `search_mast` previously had a dead `allowed_inst = ['NIRISS', 'NIRISS/IMAGE']` line immediately overridden by `['MIRI', 'MIRI/IMAGE']`, restricting MAST searches to MIRI only. Fixed to `['NIRISS', 'NIRISS/IMAGE', 'NIRCAM', 'NIRCAM/IMAGE', 'MIRI', 'MIRI/IMAGE']`. A corresponding empty-table guard was added to `search_data_products_by_obs` so that targets with no matching observations produce a clean empty result instead of crashing with `InvalidQueryError`.
    - **Robust Catalog Parsing**: Implemented regex-based fixed-width parsing for `.XYmqxyrd` generation, ensuring compatibility with varying engine outputs and preventing `ValueError` crashes on line slicing.
    - **Environment Reliability**: Transitioned all sub-tool calls to use `subprocess.run` with explicit `PYTHONPATH` inheritance and absolute paths, ensuring consistent execution across different environments.
    - **File Displacement Fix**: Resolved a critical bug in the photometry loop that was prematurely moving FITS files before ART passes could complete.
    - **Sanitization**: Removed corrupted "poison strings" that were causing `NameError` failures in core functions.
    - **Missing-Image Robustness (June 2026)**: `find_stars_to_align` used bare `return` (returning `None`) when a FITS file was absent. The caller in `launch_xym2pm_GH` assigned the result back to `Gaia_JWST_table`, silently replacing the valid catalog with `None` and causing `AttributeError: 'NoneType' object has no attribute 'loc'`. Fixed: `find_stars_to_align` now returns `stars_catalog` unchanged on missing/unreadable files. `launch_xym2pm_GH` additionally guards each image with an explicit `os.path.isfile` check and `continue` so absent images are fully skipped without entering the alignment loop.
    - **MAST Download Error Handling (June 2026)**: `download_JWST_images` had a bare `except:` around `Observations.download_products(Table.from_pandas(...))`. This was intended to handle pandas→astropy Table type differences, but it also caught `requests.exceptions.ConnectionError` (Errno 113 No route to host, typical on HPC compute nodes without internet), then retried the same network call — producing a second crash with no useful guidance. Fixed: separated the type-conversion try/except `(TypeError, ValueError)` from the download call; network errors are now caught as `(RequestsConnectionError, OSError)` and re-raised as `ConnectionError` with a message instructing the user to rerun with `--load_existing` if files are already downloaded locally.

## Known Bugs

### `gdc_filename` UnboundLocalError on re-runs (`jwst1pass_GH`)

**Symptom:** `UnboundLocalError: local variable 'gdc_filename' referenced before assignment` at the `stdgdc_data = fits.open(gdc_filename)` line inside `jwst1pass_GH`, raised via `jwst1pass_GH_multiproc_run` in the multiprocessing pool.

**Root cause:** The header-reading block that assigns `gdc_filename` (and `psf_filename`, `fmin`, etc.) sits inside `if not os.path.isfile(XYmqxyrd_filename):`. When the XYmqxyrd catalog already exists from a previous run, this block is skipped entirely. The GDC correction step that calls `fits.open(gdc_filename)` runs unconditionally, crashing because `gdc_filename` was never set.

**Workaround:** Pass `--jwst1pass` (sets `force_jwst1pass=True`), which deletes the existing XYmqxyrd before the conditional is evaluated, forcing the header-reading block to execute and `gdc_filename` to be assigned.

**Proper fix (pending):** Move the header-reading and filename-assignment block (FITS file check, instrument/detector/filter parsing, `gdc_filename`/`psf_filename` construction) to before the `if not os.path.isfile(XYmqxyrd_filename):` guard, so those variables are always defined regardless of catalog existence.

## Technical Mandates

### 1. Astrometric Projection
- **Center Point**: Always use `CRVAL` from the image extension (`hdul[1]`) for the projection center. Using `TARG_RA/DEC` from the primary header will cause large systematic shifts and alignment failure.
- **Reference Pixel**: Use the actual `CRPIX` values (typically 1024.5 for NIRISS) instead of hardcoded legacy values like 5000.0.

### 2. Error Handling
- **Missing Alignments**: If the alignment tool fails to produce a `.MAT` or `.LNK` file, the image must be explicitly skipped to avoid corrupting the final proper motion average.
- **Plotting Safety**: Statistical routines (like `bin_errors`) must initialize variables with `NaN` before `try-except` blocks to prevent crashes on sparse or problematic data.

### 3. Coordinate Parity
- **1-Based Indexing**: While Python is 0-indexed, all coordination with legacy Fortran tools MUST maintain 1-based indexing for pixel coordinates to ensure parity with the Anderson-style astrometric engine.

## Usage Conventions
- **Command Line**: Prefer using the coordinated `GaiaJWST.py` script with explicit flags (e.g., `--name`, `--ra`, `--dec`).
- **Data Cleanup**: Use `--remove_previous_files` when rerunning alignment to ensure "poisonous" legacy `.MAT` files do not corrupt new alignment attempts.
