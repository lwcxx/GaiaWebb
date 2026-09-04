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
    - **Robust Catalog Parsing**: Implemented regex-based fixed-width parsing for `.XYmqxyrd` generation, ensuring compatibility with varying engine outputs and preventing `ValueError` crashes on line slicing.
    - **Environment Reliability**: Transitioned all sub-tool calls to use `subprocess.run` with explicit `PYTHONPATH` inheritance and absolute paths, ensuring consistent execution across different environments.
    - **File Displacement Fix**: Resolved a critical bug in the photometry loop that was prematurely moving FITS files before ART passes could complete.
    - **Sanitization**: Removed corrupted "poison strings" that were causing `NameError` failures in core functions.

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
