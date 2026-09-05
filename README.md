# GaiaWebb

GaiaWebb is a Python-only tool that computes proper motions combining data from Gaia and the James Webb Space Telescope. The python code from GaiaWebb is based on GaiaHub [del Pino et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...76D/abstract). jwst1pass_py_v2 is a Python translation based on the Fortran JWST1Pass from [Anderson, 2022](https://ui.adsabs.harvard.edu/abs/2022acs..rept....2A/abstract) and [Libralato et al., 2023](https://iopscience.iop.org/article/10.3847/1538-4357/acd04f). The high-precision NIRISS point spread functions (PSFs) and geometric-distortion correction (GDC) were applied according to [Libralato et al., 2023](https://iopscience.iop.org/article/10.3847/1538-4357/acd04f). PSFs and GDC of MIRI and NIRCam were implemented using the [repository](https://www.stsci.edu/~jayander/JWST1PASS/).

## License and Referencing
GaiaHub and its documentation are released under a BSD 2-clause license. GaiaHub is freely available and you can freely modify, extend, and improve the GaiaHub source code. However, if you use it for published research, you are requested to cite [del Pino et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...76D/abstract) where the method is described.


## Features

The same as GaiaHub, GaiaWebb includes lots of useful features:

* Search of objects based on names.
* Automatic screening out of poorly measured stars.
* Interactive and automatic selection of member stars.
* Statistics about the systemic proper motions of the object.
* Automatic generation of figures.

## Running GaiaWebb

Once the installation is completed, the user can run GaiaWebb from the terminal as:

$ python /path/to/GaiaWJWST.py

For example, to compute the proper motions of NGC 5053 using only member stars, GaiaHub should be called as:

$ python /path/to/GaiaWJWST.py --name "NGC 5053" --use_members

In this example, the results produced by GaiaWebb will be stored in a subfolder called "NGC_5053".

To know more about all GaiaWebb options:

$ python /path/to/GaiaWJWST.py --help

## Notes on GaiaWebb

GaiaWebb has one more argument --instrument compared to GaiaHub. The additional argument enables the selection of JWST cameras, NIRISS, MIRI, or NIRCam. 

The Fortran JWST1Pass has been translated to Python code. Some of the main differences between the two are:

1. The Python version filters out the low-quality pixels before fitting stars while the Fortran version does the filtering during the fitting.
2. The Python version iteratively subtract neighbourhood stars, which is helpful in crowded region.
3. The Python version fits flux, position, and sky simultaneous and the Fortran version fits them separately.
4. The Python version has multi-pass to subtract stars to find fainter stars.
5. Noise map and gain map are found and downloaded for each image if available.
