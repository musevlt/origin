import datetime
import glob
import inspect
import logging
import os
import shutil
import sys
import warnings
from collections import OrderedDict
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils import lazyproperty
from mpdaf.log import setup_logging
from mpdaf.MUSE import FieldsMap
from mpdaf.obj import Cube, Image

from . import steps
from .lib_origin import timeit
from .version import __version__

try:
    # With PyYaml 5.1, load and safe have been renamed to unsafe_* and
    # replaced by the safe_* functions. We need the full ones to
    # be able to dump Python objects, yay!
    from yaml import unsafe_load as load_yaml, dump as dump_yaml
except ImportError:  # pragma: no cover
    from yaml import load as load_yaml, dump as dump_yaml

CURDIR = os.path.dirname(os.path.abspath(__file__))


class ORIGIN(steps.LogMixin):
    """ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

    This is the main class to interact with all the steps.  An Origin object is
    mainly composed by:
    - cube data (raw data and covariance)
    - 1D dictionary of spectral profiles
    - MUSE PSF

    Attributes
    ----------
    path : str
        Path where the ORIGIN data will be stored.
    name : str
        Name of the session and basename for the sources.
    param : dict
        Parameters values.
    cube_raw : array (Nz, Ny, Nx)
        Raw data.
    var : array (Nz, Ny, Nx)
        Variance.
    wcs : `mpdaf.obj.WCS`
        RA-DEC coordinates.
    wave : `mpdaf.obj.WaveCoord`
        Spectral coordinates.
    profiles : list of array
        List of spectral profiles to test
    FWHM_profiles : list
        FWHM of the profiles in pixels.
    wfields : None or list of arrays
        List of weight maps (one per fields in the case of MUSE mosaic)
        None: just one field
    PSF : array (Nz, PSF_size, PSF_size) or list of arrays
        MUSE PSF (one per field)
    LBDA_FWHM_PSF: list of floats
        Value of the FWMH of the PSF in pixel for each wavelength step (mean of
        the fields).
    FWHM_PSF : float or list of float
        Mean of the fwhm of the PSF in pixel (one per field).
    imawhite : `~mpdaf.obj.Image`
        White image
    segmap : `~mpdaf.obj.Image`
        Segmentation map
    cube_std : `~mpdaf.obj.Cube`
        standardized data for PCA. Result of step01.
    cont_dct : `~mpdaf.obj.Cube`
        DCT continuum. Result of step01.
    ima_std : `~mpdaf.obj.Image`
        Mean of standardized data for PCA along the wavelength axis.
        Result of step01.
    ima_dct : `~mpdaf.obj.Image`
        Mean of DCT continuum cube along the wavelength axis.
        Result of step01.
    nbAreas : int
        Number of area (segmentation) for the PCA computation.
        Result of step02.
    areamap : `~mpdaf.obj.Image`
        PCA area. Result of step02.
    testO2 : list of arrays (one per PCA area)
        Result of the O2 test (step03).
    histO2 : list of arrays (one per PCA area)
        PCA histogram (step03).
    binO2 : list of arrays (one per PCA area)
        Bins for the PCA histogram (step03).
    thresO2 : list of float
        For each area, threshold value (step03).
    meaO2 : list of float
        Location parameter of the Gaussian fit used to
        estimate the threshold (step03).
    stdO2 : list of float
        Scale parameter of the Gaussian fit used to
        estimate the threshold (step03).
    cube_faint : `~mpdaf.obj.Cube`
        Projection on the eigenvectors associated to the lower eigenvalues
        of the data cube (representing the faint signal). Result of step04.
    mapO2 : `~mpdaf.obj.Image`
        The numbers of iterations used by testO2 for each spaxel.
        Result of step04.
    cube_correl : `~mpdaf.obj.Cube`
        Cube of T_GLR values (step05).
    cube_profile : `~mpdaf.obj.Cube` (type int)
        PSF profile associated to the T_GLR (step05).
    maxmap : `~mpdaf.obj.Image`
        Map of maxima along the wavelength axis (step05).
    cube_local_max : `~mpdaf.obj.Cube`
        Local maxima from max correlation (step05).
    cube_local_min : `~mpdaf.obj.Cube`
        Local maxima from min correlation (step05).
    threshold : float
        Estimated threshold (step06).
    Pval : `astropy.table.Table`
        Table with the purity results for each threshold (step06):
        - PVal_r : The purity function
        - index_pval : index value to plot
        - Det_m : Number of detections (-DATA)
        - Det_M : Number of detections (+DATA)
    Cat0 : `astropy.table.Table`
        Catalog returned by step07
    Pval_comp : `astropy.table.Table`
        Table with the purity results for each threshold in compl (step08):
        - PVal_r : The purity function
        - index_pval : index value to plot
        - Det_m : Number of detections (-DATA)
        - Det_M : Number of detections (+DATA)
    Cat1 : `astropy.table.Table`
        Catalog returned by step08
    spectra : list of `~mpdaf.obj.Spectrum`
        Estimated lines. Result of step09.
    Cat2 : `astropy.table.Table`
        Catalog returned by step09.

    """

    def __init__(
        self,
        filename,
        name="origin",
        path=".",
        loglevel="DEBUG",
        logcolor=False,
        fieldmap=None,
        profiles=None,
        PSF=None,
        LBDA_FWHM_PSF=None,
        FWHM_PSF=None,
        PSF_size=25,
        param=None,
        imawhite=None,
        wfields=None,
    ):
        self.path = path
        self.name = name
        self.outpath = os.path.join(path, name)
        self.param = param or {}
        self.file_handler = None
        os.makedirs(self.outpath, exist_ok=True)

        # stdout & file logger
        setup_logging(
            name="muse_origin",
            level=loglevel,
            color=logcolor,
            fmt="%(levelname)-05s: %(message)s",
            stream=sys.stdout,
        )
        self.logger = logging.getLogger("muse_origin")
        self._setup_logfile(self.logger)
        self.param["loglevel"] = loglevel
        self.param["logcolor"] = logcolor

        self._loginfo("Step 00 - Initialization (ORIGIN v%s)", __version__)

        # dict of Step instances, indexed by step names
        self.steps = OrderedDict()
        # dict containing the data attributes of each step, to expose them on
        # the ORIGIN object
        self._dataobjs = {}
        for i, cls in enumerate(steps.STEPS, start=1):
            # Instantiate the step object, give it a step number
            step = cls(self, i, self.param)
            # force its signature to be the same as step.run (without the
            # ORIGIN instance), which allows to see its arguments and their
            # default value.
            sig = inspect.signature(step.run)
            step.__signature__ = sig.replace(
                parameters=[p for p in sig.parameters.values() if p.name != "orig"]
            )
            self.steps[step.name] = step
            # Insert the __call__ method of the step in the ORIGIN object. This
            # allows to run a step with a method like "step01_preprocessing".
            self.__dict__[step.method_name] = step
            for name, _ in step._dataobjs:
                self._dataobjs[name] = step

        # MUSE data cube
        self._loginfo("Read the Data Cube %s", filename)
        self.param["cubename"] = filename
        self.cube = Cube(filename)
        self.Nz, self.Ny, self.Nx = self.shape = self.cube.shape

        # RA-DEC coordinates
        self.wcs = self.cube.wcs
        # spectral coordinates
        self.wave = self.cube.wave

        # List of spectral profile
        if profiles is None:
            profiles = os.path.join(CURDIR, "Dico_3FWHM.fits")
        self.param["profiles"] = profiles

        # FSF
        self.param["fieldmap"] = fieldmap
        self.param["PSF_size"] = PSF_size
        self._read_fsf(
            self.cube,
            fieldmap=fieldmap,
            wfields=wfields,
            PSF=PSF,
            LBDA_FWHM_PSF=LBDA_FWHM_PSF,
            FWHM_PSF=FWHM_PSF,
            PSF_size=PSF_size,
        )

        # additional images
        self.ima_white = imawhite if imawhite else self.cube.mean(axis=0)

        self.testO2, self.histO2, self.binO2 = None, None, None

        self._loginfo("00 Done")

    def __getattr__(self, name):
        # Use __getattr__ to provide access to the steps data attributes
        # via the ORIGIN object. This will also trigger the loading of
        # the objects if needed.
        if name in self._dataobjs:
            return getattr(self._dataobjs[name], name)
        else:
            raise AttributeError(f"unknown attribute {name}")

    def __dir__(self):
        return (
            super().__dir__()
            + list(self._dataobjs.keys())
            + [o.method_name for o in self.steps.values()]
        )

    @lazyproperty
    def cube_raw(self):
        # Flux - set to 0 the Nan
        return self.cube.data.filled(fill_value=0)

    @lazyproperty
    def mask(self):
        return self.cube._mask

    @lazyproperty
    def var(self):
        # variance - set to Inf the Nan
        return self.cube.var.filled(np.inf)

    @classmethod
    def init(
        cls,
        cube,
        fieldmap=None,
        profiles=None,
        PSF=None,
        LBDA_FWHM_PSF=None,
        FWHM_PSF=None,
        PSF_size=25,
        name="origin",
        path=".",
        loglevel="DEBUG",
        logcolor=False,
    ):
        """Create a ORIGIN object.

        An Origin object is composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF
        - parameters used to segment the cube in different zones.

        Parameters
        ----------
        cube : str
            Cube FITS file name
        fieldmap : str
            FITS file containing the field map (mosaic)
        profiles : str
            FITS of spectral profiles
            If None, a default dictionary of 20 profiles is used.
        PSF : str
            Cube FITS filename containing a MUSE PSF per wavelength.
            If None, PSF are computed with a Moffat function
            (13x13 pixels, beta=2.6, fwhm1=0.76, fwhm2=0.66,
            lambda1=4750, lambda2=7000)
        LBDA_FWHM_PSF: list of float
            Value of the FWMH of the PSF in pixel for each wavelength step
            (mean of the fields).
        FWHM_PSF : list of float
            FWHM of the PSFs in pixels, one per field.
        PSF_size : int
            Spatial size of the PSF (when reconstructed from the cube header).
        name : str
            Name of this session and basename for the sources.
            ORIGIN.write() method saves the session in a folder that
            has this name. The ORIGIN.load() method will be used to
            load a session, continue it or create a new from it.
        loglevel : str
            Level for the logger (defaults to DEBUG).
        logcolor : bool
            Use color for the logger levels.

        """
        return cls(
            cube,
            path=path,
            name=name,
            fieldmap=fieldmap,
            profiles=profiles,
            PSF=PSF,
            LBDA_FWHM_PSF=LBDA_FWHM_PSF,
            FWHM_PSF=FWHM_PSF,
            PSF_size=PSF_size,
            loglevel=loglevel,
            logcolor=logcolor,
        )

    @classmethod
    @timeit
    def load(cls, folder, newname=None, loglevel=None, logcolor=None):
        """Load a previous session of ORIGIN.

        ORIGIN.write() method saves a session in a folder that has the name of
        the ORIGIN object (self.name).

        Parameters
        ----------
        folder : str
            Folder name (with the relative path) where the ORIGIN data
            have been stored.
        newname : str
            New name for this session. This parameter lets the user to load a
            previous session but continue in a new one. If None, the user will
            continue the loaded session.
        loglevel : str
            Level for the logger (by default reuse the saved level).
        logcolor : bool
            Use color for the logger levels.

        """
        path = os.path.dirname(os.path.abspath(folder))
        name = os.path.basename(folder)

        with open(f"{folder}/{name}.yaml", "r") as stream:
            param = load_yaml(stream)

        if "FWHM PSF" in param:
            FWHM_PSF = np.asarray(param["FWHM PSF"])
        else:
            FWHM_PSF = None

        if "LBDA_FWHM PSF" in param:
            LBDA_FWHM_PSF = np.asarray(param["LBDA FWHM PSF"])
        else:
            LBDA_FWHM_PSF = None

        if os.path.isfile(param["PSF"]):
            PSF = param["PSF"]
        else:
            if os.path.isfile("%s/cube_psf.fits" % folder):
                PSF = "%s/cube_psf.fits" % folder
            else:
                PSF_files = glob.glob("%s/cube_psf_*.fits" % folder)
                if len(PSF_files) == 0:
                    PSF = None
                elif len(PSF_files) == 1:
                    PSF = PSF_files[0]
                else:
                    PSF = sorted(PSF_files)
        wfield_files = glob.glob("%s/wfield_*.fits" % folder)
        if len(wfield_files) == 0:
            wfields = None
        else:
            wfields = sorted(wfield_files)

        # step0
        if os.path.isfile("%s/ima_white.fits" % folder):
            ima_white = Image("%s/ima_white.fits" % folder)
        else:
            ima_white = None

        if newname is not None:
            # copy outpath to the new path
            shutil.copytree(os.path.join(path, name), os.path.join(path, newname))
            name = newname

        loglevel = loglevel if loglevel is not None else param["loglevel"]
        logcolor = logcolor if logcolor is not None else param["logcolor"]

        obj = cls(
            path=path,
            name=name,
            param=param,
            imawhite=ima_white,
            loglevel=loglevel,
            logcolor=logcolor,
            filename=param["cubename"],
            fieldmap=param["fieldmap"],
            wfields=wfields,
            profiles=param["profiles"],
            PSF=PSF,
            FWHM_PSF=FWHM_PSF,
            LBDA_FWHM_PSF=LBDA_FWHM_PSF,
        )

        for step in obj.steps.values():
            step.load(obj.outpath)

        # special case for step3
        NbAreas = param.get("nbareas")
        if NbAreas is not None:
            if os.path.isfile("%s/testO2_1.txt" % folder):
                obj.testO2 = [
                    np.loadtxt("%s/testO2_%d.txt" % (folder, area), ndmin=1)
                    for area in range(1, NbAreas + 1)
                ]
            if os.path.isfile("%s/histO2_1.txt" % folder):
                obj.histO2 = [
                    np.loadtxt("%s/histO2_%d.txt" % (folder, area), ndmin=1)
                    for area in range(1, NbAreas + 1)
                ]
            if os.path.isfile("%s/binO2_1.txt" % folder):
                obj.binO2 = [
                    np.loadtxt("%s/binO2_%d.txt" % (folder, area), ndmin=1)
                    for area in range(1, NbAreas + 1)
                ]

        return obj

    def info(self):
        """Prints the processing log."""
        with open(self.logfile) as f:
            for line in f:
                if line.find("Done") == -1:
                    print(line, end="")

    def status(self):
        """Prints the processing status."""
        for name, step in self.steps.items():
            print(f"- {step.idx:02d}, {name}: {step.status.name}")

    def _setup_logfile(self, logger):
        if self.file_handler is not None:
            # Remove the handlers before adding a new one
            self.file_handler.close()
            logger.handlers.remove(self.file_handler)

        self.logfile = os.path.join(self.outpath, self.name + ".log")
        self.file_handler = RotatingFileHandler(self.logfile, "a", 1000000, 1)
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)

    def set_loglevel(self, level):
        """Set the logging level for the console logger."""
        handler = next(
            h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)
        )
        handler.setLevel(level)
        self.param["loglevel"] = level

    @property
    def nbAreas(self):
        """Number of area (segmentation) for the PCA."""
        return self.param.get("nbareas")

    @property
    def threshold_correl(self):
        """Estimated threshold used to detect lines on local maxima of max
        correl."""
        return self.param.get("threshold")

    @threshold_correl.setter
    def threshold_correl(self, value):
        self.param["threshold"] = value

    @property
    def threshold_std(self):
        """Estimated threshold used to detect complementary lines on local
        maxima of std cube."""
        return self.param.get("threshold_std")

    @threshold_std.setter
    def threshold_std(self, value):
        self.param["threshold_std"] = value

    @lazyproperty
    def profiles(self):
        """Read the list of spectral profiles."""
        profiles = self.param["profiles"]
        self._loginfo("Load dictionary of spectral profile %s", profiles)
        with fits.open(profiles) as hdul:
            profiles = [hdu.data for hdu in hdul[1:]]

        # check that the profiles have the same size
        if len({p.shape[0] for p in profiles}) != 1:
            raise ValueError("The profiles must have the same size")

        return profiles

    @lazyproperty
    def FWHM_profiles(self):
        """Read the list of FWHM of the spectral profiles."""
        with fits.open(self.param["profiles"]) as hdul:
            return [hdu.header["FWHM"] for hdu in hdul[1:]]

    def _read_fsf(
        self,
        cube,
        fieldmap=None,
        wfields=None,
        PSF=None,
        LBDA_FWHM_PSF=None,
        FWHM_PSF=None,
        PSF_size=25,
    ):
        """Read FSF cube(s), with fieldmap in the case of MUSE mosaic.

        There are two ways to specify the PSF informations:

        - with the ``PSF``, ``FWHM_PSF``, and ``LBDA_FWHM`` parameters.
        - or read from the cube header and fieldmap.

        If there are multiple fields, for a mosaic, we also need weight maps.
        If the cube contains a FSF model and a fieldmap is given, these weight
        maps are computed automatically.

        Parameters
        ----------
        cube : mpdaf.obj.Cube
            The input datacube.
        fieldmap : str
            FITS file containing the field map (mosaic).
        wfields : list of str
            List of weight maps (one per fields in the case of MUSE mosaic).
        PSF : str or list of str
            Cube FITS filename containing a MUSE PSF per wavelength, or a list
            of filenames for multiple fields (mosaic).
        LBDA_FWHM_PSF: list of float
            Value of the FWMH of the PSF in pixel for each wavelength step
            (mean of the fields).
        FWHM_PSF : list of float
            FWHM of the PSFs in pixels, one per field.
        PSF_size : int
            Spatial size of the PSF (when reconstructed from the cube header).

        """
        self.wfields = None
        info = self.logger.info

        if PSF is None or FWHM_PSF is None or LBDA_FWHM_PSF is None:
            info("Compute FSFs from the datacube FITS header keywords")
            if "FSFMODE" not in cube.primary_header:
                raise ValueError("missing PSF keywords in the cube FITS header")

            # FSF created from FSF*** keywords
            try:
                from mpdaf.MUSE import FSFModel
            except ImportError:
                sys.exit("you must upgrade MPDAF")

            fsf = FSFModel.read(cube)
            lbda = cube.wave.coord()
            shape = (PSF_size, PSF_size)
            if isinstance(fsf, FSFModel):  # just one FSF
                self.PSF = fsf.get_3darray(lbda, shape)
                self.LBDA_FWHM_PSF = fsf.get_fwhm(lbda, unit="pix")
                self.FWHM_PSF = np.mean(self.LBDA_FWHM_PSF)
                # mean of the fwhm of the FSF in pixel
                info("mean FWHM of the FSFs = %.2f pixels", self.FWHM_PSF)
            else:
                self.PSF = [f.get_3darray(lbda, shape) for f in fsf]
                fwhm = np.array([f.get_fwhm(lbda, unit="pix") for f in fsf])
                self.LBDA_FWHM_PSF = np.mean(fwhm, axis=0)
                self.FWHM_PSF = np.mean(fwhm, axis=1)
                for i, fwhm in enumerate(self.FWHM_PSF):
                    info("mean FWHM of the FSFs (field %d) = %.2f pixels", i, fwhm)
                info("Compute weight maps from field map %s", fieldmap)
                fmap = FieldsMap(fieldmap, nfields=len(fsf))
                # weighted field map
                self.wfields = fmap.compute_weights()

            self.param["PSF"] = cube.primary_header["FSFMODE"]
        else:
            self.LBDA_FWHM_PSF = LBDA_FWHM_PSF

            if isinstance(PSF, str):
                info("Load FSFs from %s", PSF)
                self.param["PSF"] = PSF

                self.PSF = fits.getdata(PSF)
                if self.PSF.shape[1] != self.PSF.shape[2]:
                    raise ValueError("PSF must be a square image.")
                if not self.PSF.shape[1] % 2:
                    raise ValueError("The spatial size of the PSF must be odd.")
                if self.PSF.shape[0] != self.shape[0]:
                    raise ValueError(
                        "PSF and data cube have not the same"
                        "dimensions along the spectral axis."
                    )
                # mean of the fwhm of the FSF in pixel
                self.FWHM_PSF = np.mean(FWHM_PSF)
                self.param["FWHM PSF"] = FWHM_PSF.tolist()
                info("mean FWHM of the FSFs = %.2f pixels", self.FWHM_PSF)
            else:
                nfields = len(PSF)
                self.wfields = []
                self.PSF = []
                self.FWHM_PSF = list(FWHM_PSF)

                for n in range(nfields):
                    info("Load FSF from %s", PSF[n])
                    self.PSF.append(fits.getdata(PSF[n]))
                    info("Load weight maps from %s", wfields[n])
                    self.wfields.append(fits.getdata(wfields[n]))
                    info(
                        "mean FWHM of the FSFs (field %d) = %.2f pixels", n, FWHM_PSF[n]
                    )

        self.param["FWHM PSF"] = self.FWHM_PSF.tolist()
        self.param["LBDA FWHM PSF"] = self.LBDA_FWHM_PSF.tolist()

    @timeit
    def write(self, path=None, erase=False):
        """Save the current session in a folder that will have the name of the
        ORIGIN object (self.name).

        The ORIGIN.load(folder, newname=None) method will be used to load a
        session. The parameter newname will let the user to load a session but
        continue in a new one.

        Parameters
        ----------
        path : str
            Path where the folder (self.name) will be stored.
        erase : bool
            Remove the folder if it exists.

        """
        self._loginfo("Writing...")

        # adapt session if path changes
        if path is not None and path != self.path:
            if not os.path.exists(path):
                raise ValueError(f"path does not exist: {path}")
            self.path = path
            outpath = os.path.join(path, self.name)
            # copy outpath to the new path
            shutil.copytree(self.outpath, outpath)
            self.outpath = outpath
            self._setup_logfile(self.logger)

        if erase:
            shutil.rmtree(self.outpath)
        os.makedirs(self.outpath, exist_ok=True)

        # PSF
        if isinstance(self.PSF, list):
            for i, psf in enumerate(self.PSF):
                cube = Cube(data=psf, mask=np.ma.nomask, copy=False)
                cube.write(os.path.join(self.outpath, "cube_psf_%02d.fits" % i))
        else:
            cube = Cube(data=self.PSF, mask=np.ma.nomask, copy=False)
            cube.write(os.path.join(self.outpath, "cube_psf.fits"))

        if self.wfields is not None:
            for i, wfield in enumerate(self.wfields):
                im = Image(data=wfield, mask=np.ma.nomask)
                im.write(os.path.join(self.outpath, "wfield_%02d.fits" % i))

        if self.ima_white is not None:
            self.ima_white.write("%s/ima_white.fits" % self.outpath)

        for step in self.steps.values():
            step.dump(self.outpath)

        # parameters in .yaml
        with open(f"{self.outpath}/{self.name}.yaml", "w") as stream:
            dump_yaml(self.param, stream)

        # step3 - saving this manually for now
        if self.nbAreas is not None:
            if self.testO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt(
                        "%s/testO2_%d.txt" % (self.outpath, area), self.testO2[area - 1]
                    )
            if self.histO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt(
                        "%s/histO2_%d.txt" % (self.outpath, area), self.histO2[area - 1]
                    )
            if self.binO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt(
                        "%s/binO2_%d.txt" % (self.outpath, area), self.binO2[area - 1]
                    )

        self._loginfo("Current session saved in %s", self.outpath)

    def plot_areas(self, ax=None, **kwargs):
        """ Plot the 2D segmentation for PCA from self.step02_areas()
        on the test used to perform this segmentation.

        Parameters
        ----------
        ax : matplotlib.Axes
            The Axes instance in which the image is drawn.
        kwargs : matplotlib.artist.Artist
            Optional extra keyword/value arguments to be passed to ``ax.imshow()``.

        """
        if ax is None:
            ax = plt.gca()

        kwargs.setdefault("cmap", "jet")
        kwargs.setdefault("alpha", 0.7)
        kwargs.setdefault("interpolation", "nearest")
        kwargs["origin"] = "lower"
        cax = ax.imshow(self.areamap._data, **kwargs)

        i0 = np.min(self.areamap._data)
        i1 = np.max(self.areamap._data)
        if i0 != i1:
            from matplotlib.colors import BoundaryNorm
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            n = i1 - i0 + 1
            bounds = np.linspace(i0, i1 + 1, n + 1) - 0.5
            norm = BoundaryNorm(bounds, n + 1)
            divider = make_axes_locatable(ax)
            cax2 = divider.append_axes("right", size="5%", pad=1)
            plt.colorbar(
                cax,
                cax=cax2,
                cmap=kwargs["cmap"],
                norm=norm,
                spacing="proportional",
                ticks=bounds + 0.5,
                boundaries=bounds,
                format="%1i",
            )

    def plot_step03_PCA_threshold(
        self, log10=False, ncol=3, legend=True, xlim=None, fig=None, **fig_kw
    ):
        """ Plot the histogram and the threshold for the starting point of the PCA.

        Parameters
        ----------
        log10 : bool
            Draw histogram in logarithmic scale or not
        ncol : int
            Number of colomns in the subplots
        legend : bool
            If true, write pfa and threshold values as legend
        xlim : (float, float)
            Set the data limits for the x-axes
        fig : matplotlib.Figure
            Figure instance in which the image is drawn
        **fig_kw : matplotlib.artist.Artist
            All additional keyword arguments are passed to the figure() call.

        """
        if self.nbAreas is None:
            raise ValueError("Run the step 02 to initialize self.nbAreas")

        if fig is None:
            fig = plt.figure()

        if self.nbAreas <= ncol:
            n = 1
            m = self.nbAreas
        else:
            n = self.nbAreas // ncol
            m = ncol
            if (n * m) < self.nbAreas:
                n = n + 1

        for area in range(1, self.nbAreas + 1):
            if area == 1:
                ax = fig.add_subplot(n, m, area, **fig_kw)
            else:
                ax = fig.add_subplot(n, m, area, sharey=fig.axes[0], **fig_kw)
            self.plot_PCA_threshold(area, "step03", log10, legend, xlim, ax)

        # Fine-tune figure
        for a in fig.axes[:-1]:
            a.set_xlabel("")
        for a in fig.axes[1:]:
            a.set_ylabel("")
        plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
        plt.setp([a.get_yticklabels() for a in fig.axes[0::m]], visible=True)
        plt.setp([a.get_yticklines() for a in fig.axes], visible=False)
        plt.setp([a.get_yticklines() for a in fig.axes[0::m]], visible=True)
        fig.subplots_adjust(wspace=0)
        if xlim is not None:
            plt.setp([a.get_xticklabels() for a in fig.axes[:-m]], visible=False)
            plt.setp([a.get_xticklines() for a in fig.axes[:-m]], visible=False)
            fig.subplots_adjust(hspace=0)

    def plot_step03_PCA_stat(self, cutoff=5, ax=None):
        """Plot the threshold value according to the area.

        Median Absolute Deviation is used to find outliers.

        Parameters
        ----------
        cutoff : float
            Median Absolute Deviation cutoff
        ax : matplotlib.Axes
            The Axes instance in which the image is drawn

        """
        if self.nbAreas is None:
            raise ValueError("Run the step 02 to initialize self.nbAreas")
        if self.thresO2 is None:
            raise ValueError("Run the step 03 to compute the threshold values")
        if ax is None:
            ax = plt.gca()
        ax.plot(np.arange(1, self.nbAreas + 1), self.thresO2, "+")
        med = np.median(self.thresO2)
        diff = np.absolute(self.thresO2 - med)
        mad = np.median(diff)
        if mad != 0:
            ksel = (diff / mad) > cutoff
            if ksel.any():
                ax.plot(
                    np.arange(1, self.nbAreas + 1)[ksel],
                    np.asarray(self.thresO2)[ksel],
                    "ro",
                )
        ax.set_xlabel("area")
        ax.set_ylabel("Threshold")
        ax.set_title(f"PCA threshold (med={med:.2f}, mad= {mad:.2f})")

    def plot_PCA_threshold(
        self, area, pfa_test="step03", log10=False, legend=True, xlim=None, ax=None
    ):
        """ Plot the histogram and the threshold for the starting point of the PCA.

        Parameters
        ----------
        area : int in [1, nbAreas]
            Area ID
        pfa_test : float or str
            PFA of the test (if 'step03', the value set during step03 is used)
        log10 : bool
            Draw histogram in logarithmic scale or not
        legend : bool
            If true, write pfa and threshold values as legend
        xlim : (float, float)
            Set the data limits for the x-axis
        ax : matplotlib.Axes
            Axes instance in which the image is drawn

        """
        if self.nbAreas is None:
            raise ValueError("Run the step 02 to initialize self.nbAreas")

        if pfa_test == "step03":
            param = self.param["compute_PCA_threshold"]["params"]
            if "pfa_test" in param:
                pfa_test = param["pfa_test"]
                hist = self.histO2[area - 1]
                bins = self.binO2[area - 1]
                thre = self.thresO2[area - 1]
                mea = self.meaO2[area - 1]
                std = self.stdO2[area - 1]
            else:
                raise ValueError(
                    "pfa_test param is None: set a value or run the Step03"
                )
        else:
            if self.cube_std is None:
                raise ValueError("Run the step 01 to initialize self.cube_std")
            # limits of each spatial zone
            ksel = self.areamap._data == area
            # Data in this spatio-spectral zone
            cube_temp = self.cube_std._data[:, ksel]
            # Compute_PCA_threshold
            from .lib_origin import Compute_PCA_threshold

            testO2, hist, bins, thre, mea, std = Compute_PCA_threshold(
                cube_temp, pfa_test
            )

        if ax is None:
            ax = plt.gca()

        from scipy import stats

        center = (bins[:-1] + bins[1:]) / 2
        gauss = stats.norm.pdf(center, loc=mea, scale=std)
        gauss *= hist.max() / gauss.max()

        if log10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gauss = np.log10(gauss)
                hist = np.log10(hist)

        ax.plot(center, hist, "-k")
        ax.plot(center, hist, ".r")
        ax.plot(center, gauss, "-b", alpha=0.5)
        ax.axvline(thre, color="b", lw=2, alpha=0.5)
        ax.grid()
        if xlim is None:
            ax.set_xlim((center.min(), center.max()))
        else:
            ax.set_xlim(xlim)
        ax.set_xlabel("frequency")
        ax.set_ylabel("value")
        kwargs = dict(transform=ax.transAxes, bbox=dict(facecolor="red", alpha=0.5))
        if legend:
            text = "zone %d\npfa %.2f\nthreshold %.2f" % (area, pfa_test, thre)
            ax.text(0.1, 0.8, text, **kwargs)
        else:
            ax.text(0.9, 0.9, "%d" % area, **kwargs)

    def plot_mapPCA(self, area=None, iteration=None, ax=None, **kwargs):
        """ Plot at a given iteration (or at the end) the number of times
        a spaxel got cleaned by the PCA.

        Parameters
        ----------
        area: int in [1, nbAreas]
            if None draw the full map for all areas
        iteration : int
            Display the nuisance/bacground pixels at iteration k
        ax : matplotlib.Axes
            The Axes instance in which the image is drawn
        kwargs : matplotlib.artist.Artist
            Optional extra keyword/value arguments to be passed to ``ax.imshow()``.

        """
        if self.mapO2 is None:
            raise ValueError("Run the step 04 to initialize self.mapO2")

        themap = self.mapO2.copy()
        title = "Number of times the spaxel got cleaned by the PCA"
        if iteration is not None:
            title += "\n%d iterations" % iteration
        if area is not None:
            mask = np.ones_like(self.mapO2._data, dtype=np.bool)
            mask[self.areamap._data == area] = False
            themap._mask = mask
            title += " (zone %d)" % area

        if iteration is not None:
            themap[themap._data < iteration] = np.ma.masked

        if ax is None:
            ax = plt.gca()

        kwargs.setdefault("cmap", "jet")
        themap.plot(title=title, colorbar="v", ax=ax, **kwargs)

    def plot_purity(self, comp=False, ax=None, log10=False, legend=True):
        """Draw number of sources per threshold computed in step06/step08.

        Parameters
        ----------
        comp : bool
            If True, plot purity curves for the complementary lines (step08).
        ax : matplotlib.Axes
            The Axes instance in which the image is drawn.
        log10 : bool
            To draw histogram in logarithmic scale or not.
        legend : bool
            To draw the legend.

        """
        if ax is None:
            ax = plt.gca()

        if comp:
            threshold = self.threshold_std
            purity = self.param["purity_std"]
            Pval = self.Pval_comp
        else:
            threshold = self.threshold_correl
            purity = self.param["purity"]
            Pval = self.Pval

        if Pval is None:
            raise ValueError("Run the step 06")

        Tval_r = Pval["Tval_r"]
        ax2 = ax.twinx()
        ax2.plot(Tval_r, Pval["Pval_r"], "y.-", label="purity")
        ax.plot(Tval_r, Pval["Det_M"], "b.-", label="n detections (+DATA)")
        ax.plot(Tval_r, Pval["Det_m"], "g.-", label="n detections (-DATA)")
        ax2.plot(threshold, purity, "xr")
        if log10:
            ax.set_yscale("log")
            ax2.set_yscale("log")

        ym, yM = ax.get_ylim()
        ax.plot(
            [threshold, threshold],
            [ym, yM],
            "r",
            alpha=0.25,
            lw=2,
            label="automatic threshold",
        )

        ax.set_ylim((ym, yM))
        ax.set_xlabel("Threshold")
        ax2.set_ylabel("Purity")
        ax.set_ylabel("Number of detections")
        ax.set_title("threshold %f" % threshold)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if legend:
            ax.legend(h1 + h2, l1 + l2, loc=2)

    def plot_NB(self, src_ind, ax1=None, ax2=None, ax3=None):
        """Plot the narrow band images.

        Parameters
        ----------
        src_ind : int
            Index of the object in self.Cat0.
        ax1 : matplotlib.Axes
            The Axes instance in which the NB image around the source is drawn.
        ax2 : matplotlib.Axes
            The Axes instance in which a other NB image for check is drawn.
        ax3 : matplotlib.Axes
            The Axes instance in which the difference is drawn.

        """
        if self.Cat0 is None:
            raise ValueError("Run the step 05 to initialize self.Cat0")

        if ax1 is None and ax2 is None and ax3 is None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        # Coordinates of the source
        x0 = self.Cat0[src_ind]["x0"]
        y0 = self.Cat0[src_ind]["y0"]
        z0 = self.Cat0[src_ind]["z0"]
        # Larger spatial ranges for the plots
        longxy0 = 20
        y01 = max(0, y0 - longxy0)
        y02 = min(self.shape[1], y0 + longxy0 + 1)
        x01 = max(0, x0 - longxy0)
        x02 = min(self.shape[2], x0 + longxy0 + 1)
        # Coordinates in this window
        y00 = y0 - y01
        x00 = x0 - x01
        # spectral profile
        num_prof = self.Cat0[src_ind]["profile"]
        profil0 = self.profiles[num_prof]
        # length of the spectral profile
        profil1 = profil0[profil0 > 1e-13]
        long0 = profil1.shape[0]
        # half-length of the spectral profile
        longz = long0 // 2
        # spectral range
        intz1 = max(0, z0 - longz)
        intz2 = min(self.shape[0], z0 + longz + 1)
        # subcube for the plot
        cube_test_plot = self.cube_raw[intz1:intz2, y01:y02, x01:x02]
        wcs = self.wcs[y01:y02, x01:x02]
        # controle cube
        nb_ranges = 3
        if (z0 + longz + nb_ranges * long0) < self.shape[0]:
            intz1c = intz1 + nb_ranges * long0
            intz2c = intz2 + nb_ranges * long0
        else:
            intz1c = intz1 - nb_ranges * long0
            intz2c = intz2 - nb_ranges * long0
        cube_controle_plot = self.cube_raw[intz1c:intz2c, y01:y02, x01:x02]
        # (1/sqrt(2)) * difference of the 2 sububes
        diff_cube_plot = (1 / np.sqrt(2)) * (cube_test_plot - cube_controle_plot)

        if ax1 is not None:
            ax1.plot(x00, y00, "m+")
            ima_test_plot = Image(data=cube_test_plot.sum(axis=0), wcs=wcs)
            title = "cube test - (%d,%d)\n" % (x0, y0)
            title += "lambda=%d int=[%d,%d[" % (z0, intz1, intz2)
            ima_test_plot.plot(colorbar="v", title=title, ax=ax1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

        if ax2 is not None:
            ax2.plot(x00, y00, "m+")
            ima_controle_plot = Image(data=cube_controle_plot.sum(axis=0), wcs=wcs)
            title = "check - (%d,%d)\n" % (x0, y0) + "int=[%d,%d[" % (intz1c, intz2c)
            ima_controle_plot.plot(colorbar="v", title=title, ax=ax2)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

        if ax3 is not None:
            ax3.plot(x00, y00, "m+")
            ima_diff_plot = Image(data=diff_cube_plot.sum(axis=0), wcs=wcs)
            title = "Difference narrow band - (%d,%d)\n" % (x0, y0) + "int=[%d,%d[" % (
                intz1c,
                intz2c,
            )
            ima_diff_plot.plot(colorbar="v", title=title, ax=ax3)
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)

    def plot_sources(
        self, x, y, circle=False, vmin=0, vmax=30, title=None, ax=None, **kwargs
    ):
        """Plot detected emission lines on the 2D map of maximum of the T_GLR
        values over the spectral channels.

        Parameters
        ----------
        x : array
            Coordinates along the x-axis of the estimated lines in pixels.
        y : array
            Coordinates along the y-axis of the estimated lines in pixels.
        circle : bool
            If true, plot circles with a diameter equal to the
            mean of the fwhm of the PSF.
        vmin : float
            Minimum pixel value to use for the scaling.
        vmax : float
            Maximum pixel value to use for the scaling.
        title : str
            An optional title for the figure (None by default).
        ax : matplotlib.Axes
            the Axes instance in which the image is drawn
        kwargs : matplotlib.artist.Artist
            Optional arguments passed to ``ax.imshow()``.

        """
        if ax is None:
            ax = plt.gca()
        self.maxmap.plot(vmin=vmin, vmax=vmax, title=title, ax=ax, **kwargs)

        if circle:
            fwhm = (
                self.FWHM_PSF
                if self.wfields is None
                else np.max(np.array(self.FWHM_PSF))
            )
            radius = np.round(fwhm / 2)
            for pos in zip(x, y):
                ax.add_artist(plt.Circle(pos, radius, color="k", fill=False))
        else:
            ax.plot(x, y, "k+")

    def plot_segmaps(self, axes=None, figsize=(6, 6)):
        """Plot the segmentation maps:

        - segmap_cont: segmentation map computed on the white-light image.
        - segmap_merged: segmentation map merged with the cont one and another
          one computed on the residual.
        - segmap_purity: combines self.segmap and a segmentation on the maxmap.
        - segmap_label: segmentation map used for the catalog, either the one
          given as input, otherwise self.segmap_cont.

        """
        segmaps = {}
        ncolors = 0
        for name in ("segmap_cont", "segmap_merged", "segmap_purity", "segmap_label"):
            segm = getattr(self, name, None)
            if segm:
                segmaps[name] = segm
                ncolors = max(ncolors, len(np.unique(segm._data)))

        nseg = len(segmaps)
        if nseg == 0:
            self.logger.warning("nothing to plot")
            return

        try:
            # TODO: this will be renamed to make_random_cmap in a future
            # version of photutils
            from photutils.utils.colormaps import random_cmap
        except ImportError:
            self.logger.error("photutils is needed for this")
            cmap = "jet"
        else:
            cmap = random_cmap(ncolors=ncolors)
            cmap.colors[0] = (0.0, 0.0, 0.0)

        if axes is None:
            figsize = (figsize[0] * nseg, figsize[1])
            fig, axes = plt.subplots(1, nseg, sharex=True, sharey=True, figsize=figsize)
        if nseg == 1:
            axes = [axes]

        for ax, (name, im) in zip(axes, segmaps.items()):
            im.plot(ax=ax, cmap=cmap, title=name, colorbar="v")

    def plot_min_max_hist(self, ax=None, comp=False):
        """Plot the histograms of local maxima and minima."""
        if comp:
            cube_local_max = self.cube_std_local_max._data
            cube_local_min = self.cube_std_local_min._data
        else:
            cube_local_max = self.cube_local_max._data
            cube_local_min = self.cube_local_min._data

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.set_yscale("log")
        ax.grid(which="major", linewidth=1)
        ax.grid(which="minor", linewidth=1, linestyle=":")

        maxloc = cube_local_max[cube_local_max > 0]
        bins = np.arange((maxloc.max() + 1) * 2) / 2
        ax.hist(
            maxloc, bins=bins, histtype="step", label="max", linewidth=2, cumulative=-1
        )

        minloc = cube_local_min[cube_local_min > 0]
        bins = np.arange((minloc.max() + 1) * 2) / 2
        ax.hist(
            minloc, bins=bins, histtype="step", label="min", linewidth=2, cumulative=-1
        )

        minloc2 = cube_local_min[:, self.segmap_purity._data == 0]
        minloc2 = minloc2[minloc2 > 0]
        ax.hist(
            minloc2,
            bins=bins,
            histtype="step",
            label="min filt",
            linewidth=2,
            cumulative=-1,
        )

        ax.legend()
        ax.set_title("Cumulative histogram of min/max loc")

    def timestat(self, table=False):
        """Print CPU usage by steps.

        If ``table`` is True, an astropy.table.Table is returned.

        """
        if table:
            name = []
            exdate = []
            extime = []
            tot = 0
            for s in self.steps.items():
                if "execution_date" in s[1].meta.keys():
                    name.append(s[1].method_name)
                    exdate.append(s[1].meta["execution_date"])
                    t = s[1].meta["runtime"]
                    tot += t
                    extime.append(datetime.timedelta(seconds=t))
            name.append("Total")
            exdate.append("")
            extime.append(str(datetime.timedelta(seconds=tot)))
            return Table(
                data=[name, exdate, extime],
                names=["Step", "Exec Date", "Exec Time"],
                masked=True,
            )
        else:
            tot = 0
            for s in self.steps.items():
                name = s[1].method_name
                if "execution_date" in s[1].meta.keys():
                    exdate = s[1].meta["execution_date"]
                    t = s[1].meta["runtime"]
                    tot += t
                    extime = datetime.timedelta(seconds=t)
                    self.logger.info(
                        "%s executed: %s run time: %s", name, exdate, str(extime)
                    )
            self.logger.info(
                "*** Total run time: %s", str(datetime.timedelta(seconds=tot))
            )

    def stat(self):
        """Print detection summary."""
        d = self._get_stat()
        self.logger.info(
            "ORIGIN PCA pfa %.2f Back Purity: %.2f "
            "Threshold: %.2f Bright Purity %.2f Threshold %.2f",
            d["pca"],
            d["back_purity"],
            d["back_threshold"],
            d["bright_purity"],
            d["bright_threshold"],
        )
        self.logger.info("Nb of detected lines: %d", d["tot_nlines"])
        self.logger.info(
            "Nb of sources Total: %d Background: %d Cont: %d",
            d["tot_nsources"],
            d["back_nsources"],
            d["cont_nsources"],
        )
        self.logger.info(
            "Nb of sources detected in faint (after PCA): %d "
            "in std (before PCA): %d",
            d["faint_nsources"],
            d["bright_nsources"],
        )

    def _get_stat(self):
        p = self.param
        cat = self.Cat3_sources
        if cat:
            back = cat[cat["seg_label"] == 0]
            cont = cat[cat["seg_label"] > 0]
            bright = cat[cat["comp"] == 1]
            faint = cat[cat["comp"] == 0]

        return dict(
            pca=p["compute_PCA_threshold"]["params"]["pfa_test"],
            back_purity=p["purity"],
            back_threshold=p["threshold"],
            bright_purity=p["purity_std"],
            bright_threshold=p["threshold_std"],
            tot_nlines=len(self.Cat3_lines),
            tot_nsources=len(cat),
            back_nsources=len(back),
            cont_nsources=len(cont),
            faint_nsources=len(faint),
            bright_nsources=len(bright),
        )
