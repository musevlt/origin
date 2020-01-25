"""Source file creation code."""

import logging
import os
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from joblib import Parallel, delayed
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect.source import Source
from mpdaf.tools import progressbar

import warnings
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter('ignore', category=VerifyWarning)

from .version import __version__ as origin_version

__all__ = ("create_source", "create_all_sources")


def create_source(
    source_id,
    source_table,
    source_lines,
    origin_params,
    cube_cor_filename,
    cube_std_filename,
    mask_filename,
    skymask_filename,
    spectra_fits_filename,
    segmaps,
    version,
    source_ts,
    profile_fwhm,
    *,
    author="",
    nb_fwhm=2,
    expmap_filename=None,
    save_to=None,
):
    """Create a MPDAF source.

    This function create a MPDAF source object for the ORIGIN source.

    Parameters
    ----------
    source_id : int
        Identifier for the source in the source and line tables.
    source_table : astropy.table.Table
        Catalogue of sources like the Cat3_sources one.
    source_lines : astropy.table.Table
        Catalogue of lines like the Cat3_lines one.
    origin_params : dict
        Dictionary of the parameters for the ORIGIN run.
    cube_cor_filename : str
        Name of the file containing the correlation cube of the ORIGIN run.
    cube_std_filename : str
        Name of the file containing the std cube of the ORIGIN run.
    mask_filenam e: str
        Name of the file containing the mask of the source.
    skymask_filename : str
        Name of the file containing the sky mask of the source.
    spectra_fits_filename : str
        Name of the FITS file containing the spectra of the lines.
    segmaps : dict(str: str)
        Dictionnary associating to a segmap type the associated FITS file name.
    version : str
        Version number stored in the source.
    source_ts : str
        Time stamp for when the source was created.
    profile_fwhm : list of int
        List of line profile FWHM in pixel. The index in the list is the
        profile number.
    author : str
        Name of the author.
    nb_fwhm : float
        Factor multiplying the FWHM of the line to compute the width of the
        narrow band image.
    expmap_filename : str
        Name of the file containing the exposure map.  If not None, a cut-out
        of the exposure map will be added to the source file.
    save_to : str
        If not None, the source will be saved to the given file.

    Returns
    -------
    mpdaf.sdetect.Source or None
        If save_to is used, the function returns None.

    """
    logger = logging.getLogger(__name__)

    # [0] is to get a Row not a table.
    source_info = source_table[source_table["ID"] == source_id][0]

    # The mask size is used for the cut-out size.
    mask = Image(mask_filename)
    mask_size = mask.shape[0]

    data_cube = Cube(origin_params["cubename"], convert_float64=False)

    origin = (
        "ORIGIN",
        origin_version,
        os.path.basename(origin_params["cubename"]),
        data_cube.primary_header.get("CUBE_V", ""),
    )
    source = Source.from_data(
        source_info["ID"], source_info["ra"], source_info["dec"], origin
    )

    # Information about the source in the headers
    source.header["SRC_V"] = version, "Source version"
    source.header["SRC_TS"] = source_ts, "Timestamp of the source creation"
    source.header["CAT3_TS"] = (
        source_table.meta["CAT3_TS"],
        "Timestamp of the catalog creation",
    )
    source.add_history("Source created with ORIGIN", author)

    source.header["OR_X"] = source_info["x"], "x position in pixels"
    source.header["OR_Y"] = source_info["y"], "y position in pixels"
    source.header["OR_SEG"] = (
        source_info["seg_label"],
        "Label in the segmentation map",
    )
    source.header["OR_V"] = origin_version, "ORIGIN version"
    source.header["OR_FLUX"] = source_info["flux"], "flux maximum in all lines"
    source.header["OR_PMAX"] = (source_info["purity"], "maximum purity in all lines")

    if not np.isnan(source_info["STD"]):
        source.header["OR_STD"] = (source_info["STD"], "STD max value in all lines")

    if not np.isnan(source_info["nsigSTD"]):
        source.header["OR_nSTD"] = (
            source_info["nsigSTD"],
            "max of STD/std(STD) in all lines",
        )

    if not np.isnan(source_info["T_GLR"]):
        source.header["OR_TGLR"] = (
            source_info["T_GLR"],
            "T_GLR max value in all lines",
        )
    if not np.isnan(source_info["nsigTGLR"]):
        source.header["OR_nTGLR"] = (
            source_info["nsigTGLR"],
            "max of T_GLR/std(T_GLR) in all lines",
        )

    # source_header_keyword: (key_in_origin_param, description)
    parameters_to_add = {
        "OR_PROF": ("profiles", "OR input, spectral profiles"),
        "OR_FSF": ("PSF", "OR input, FSF cube"),
        "OR_THL%02d": ("threshold_list", "OR input threshold per area"),
        "OR_NA": ("nbareas", "OR number of areas"),
        "preprocessing": {"OR_DCT": ("dct_order", "OR input, DCT order")},
        "areas": {
            "OR_PFAA": ("pfa", "OR input, PFA used to create the area map"),
            "OR_SIZA": ("maxsize", "OR input, maximum area size in pixels"),
            "OR_MSIZA": ("minsize", "OR input, minimum area size in pixels"),
        },
        "compute_PCA_threshold": {"OR_PFAT": ("pfa_test", "OR input, PFA test")},
        "compute_greedy_PCA": {
            "OR_FBG": ("Noise_population", "OR input: fraction of spectra estimated"),
            "OR_ITMAX": ("itermax", "OR input, maximum number of iterations"),
        },
        "compute_TGLR": {"OR_NG": ("size", "OR input, connectivity size")},
        "detection": {
            "OR_DXY": ("tol_spat", "OR input, spatial tolerance for merging (pix)"),
            "OR_DZ": ("tol_spec", "OR input, spectral tolerance for merging (pix)"),
        },
        "compute_spectra": {"OR_NXZ": ("grid_dxy", "OR input, grid Nxy")},
    }

    def add_keyword(keyword, param, description, params):
        if param == "threshold_list" and param in params:
            for idx, threshold in enumerate(params["threshold_list"]):
                source.header[keyword % idx] = (float("%0.2f" % threshold), description)
        elif param in params:
            if params[param] is None:
                source.header[keyword] = "", description
            else:
                source.header[keyword] = params[param], description
        else:
            logger.debug("Parameter %s absent of the parameter list.", param)

    for keyword, val in parameters_to_add.items():
        if isinstance(val, dict) and keyword in origin_params:
            for key, val2 in val.items():
                add_keyword(key, *val2, origin_params[keyword]["params"])
        else:
            add_keyword(keyword, *val, origin_params)

    source.header["COMP_CAT"] = (
        source_info["comp"],
        "1/0 (1=Pre-detected in STD, 0=detected in CORREL)",
    )

    if source.COMP_CAT:
        threshold_keyword, purity_keyword = "threshold_std", "purity_std"
    else:
        threshold_keyword, purity_keyword = "threshold", "purity"
    source.header["OR_TH"] = (
        float("%0.2f" % origin_params[threshold_keyword]),
        "OR input, threshold",
    )
    source.header["OR_PURI"] = (
        float("%0.2f" % origin_params[purity_keyword]),
        "OR input, purity",
    )

    # Mini-cubes
    source.add_cube(
        data_cube, "MUSE_CUBE", size=mask_size, unit_size=None, add_white=True
    )
    # Add FSF with the full cube, to have the same shape as fieldmap, then we
    # can work directly with the subcube
    has_fsf = True
    try:
        source.add_FSF(data_cube, fieldmap=origin_params["fieldmap"])
    except:
        logger.debug('No FSF information found in the cube')
        has_fsf = False
    data_cube = source.cubes["MUSE_CUBE"]

    if source.COMP_CAT:
        cube_ori = Cube(cube_std_filename, convert_float64=False)
        source.add_cube(cube_ori, "ORI_SNCUBE", size=mask_size, unit_size=None)
        cube_ori = source.cubes["ORI_SNCUBE"]
    else:
        cube_ori = Cube(cube_cor_filename, convert_float64=False)
        source.add_cube(cube_ori, "ORI_CORREL", size=mask_size, unit_size=None)
        cube_ori = source.cubes["ORI_CORREL"]

    # Table of sources around the exported sources.
    radius = mask_size / 2
    x_min, x_max = source_info["x"] - radius, source_info["x"] + radius
    y_min, y_max = source_info["y"] - radius, source_info["y"] + radius
    nearby_sources = (
        (source_table["x"] >= x_min)
        & (source_table["x"] <= x_max)
        & (source_table["y"] >= y_min)
        & (source_table["y"] <= y_max)
    )
    source.tables["ORI_CAT"] = source_table["ID", "ra", "dec"][nearby_sources]

    # Maps
    # The white map was added when adding the MUSE cube.
    source.images["ORI_MAXMAP"] = cube_ori.max(axis=0)
    # Using add_image, the image size is taken from the white map.
    source.add_image(mask, "ORI_MASK_OBJ")
    source.add_image(Image(skymask_filename), "ORI_MASK_SKY")
    for segmap_type, segmap_filename in segmaps.items():
        source.add_image(Image(segmap_filename), "ORI_SEGMAP_%s" % segmap_type)
    if expmap_filename is not None:
        source.add_image(Image(expmap_filename), "EXPMAP")

    # Full source spectra
    source.extract_spectra(
        data_cube, obj_mask="ORI_MASK_OBJ", sky_mask="ORI_MASK_SKY", skysub=True
    )
    source.extract_spectra(
        data_cube, obj_mask="ORI_MASK_OBJ", sky_mask="ORI_MASK_SKY", skysub=False
    )
    if source.COMP_CAT:
        source.spectra["ORI_CORR"] = (
            source.cubes["ORI_SNCUBE"] * source.images["ORI_MASK_OBJ"]
        ).mean(axis=(1, 2))
    else:
        source.spectra["ORI_CORR"] = (
            source.cubes["ORI_CORREL"] * source.images["ORI_MASK_OBJ"]
        ).mean(axis=(1, 2))

    # Add the FSF information to the source and use this information to compute
    # the PSF weighted spectra.
    if has_fsf:
        a, b, beta, _ = source.get_FSF()
        fwhm_fsf = b * data_cube.wave.coord() + a
        source.extract_spectra(
            data_cube,
            obj_mask="ORI_MASK_OBJ",
            sky_mask="ORI_MASK_SKY",
            skysub=True,
            psf=fwhm_fsf,
            beta=beta,
        )
        source.extract_spectra(
            data_cube,
            obj_mask="ORI_MASK_OBJ",
            sky_mask="ORI_MASK_SKY",
            skysub=False,
            psf=fwhm_fsf,
            beta=beta,
        )

    # Per line data: the line table, the spectrum of each line, the narrow band
    # map from the data and from the correlation cube.
    # Content of the line table in the source
    line_columns, line_units, line_fmt = zip(
        *[
            ("NUM_LINE", None, None),
            ("RA_LINE", u.deg, ".2f"),
            ("DEC_LINE", u.deg, ".2f"),
            ("LBDA_OBS", u.Angstrom, ".2f"),
            ("FWHM", u.Angstrom, ".2f"),
            ("FLUX", u.erg / (u.s * u.cm ** 2), ".1f"),
            ("GLR", None, ".1f"),
            ("nGLR", None, ".1f"),
            ("PROF", None, None),
            ("PURITY", None, ".2f"),
        ]
    )

    # If the line is a complementary one, the GLR column is replace by STD
    if source.COMP_CAT:
        line_columns = list(line_columns)
        line_columns[6] = "STD"
        line_columns[7] = "nSTD"

    # We put all the ORIGIN lines in an ORI_LINES tables but keep only the
    # unique lines in the LINES tables.
    source.add_table(source_lines, "ORI_LINES", select_in=None, col_dist=None)

    # Table containing the information on the narrow band images.
    nb_par_rows = []

    hdulist = fits.open(spectra_fits_filename)

    for line in source_lines[source_lines["merged_in"] == -9999]:
        num_line, lbda_ori, prof = line[["num_line", "lbda", "profile"]]
        fwhm_ori = profile_fwhm[prof] * data_cube.wave.get_step(unit=u.Angstrom)
        if source.COMP_CAT:
            glr_std = line["STD"]
            nglr_std = line["nsigSTD"]
        else:
            glr_std = line["T_GLR"]
            nglr_std = line["nsigTGLR"]

        source.add_line(
            cols=line_columns,
            values=[
                num_line,
                line["ra"],
                line["dec"],
                lbda_ori,
                fwhm_ori,
                line["flux"],
                glr_std,
                nglr_std,
                prof,
                line["purity"],
            ],
            units=line_units,
            fmt=line_fmt,
            desc=None,
        )

        if f"DATA{num_line}" in hdulist:  # RB add test
            source.spectra[f"ORI_SPEC_{num_line}"] = Spectrum(
                hdulist=hdulist,
                ext=(f"DATA{num_line}", f"STAT{num_line}"),
                convert_float64=False,
            )

        source.add_narrow_band_image_lbdaobs(
            data_cube,
            f"NB_LINE_{num_line}",
            lbda=lbda_ori,
            width=nb_fwhm * fwhm_ori,
            method="sum",
            subtract_off=True,
            margin=10.0,
            fband=3.0,
        )

        nb_par_rows.append(
            [f"NB_LINE_{num_line}", lbda_ori, nb_fwhm * fwhm_ori, 10.0, 3.0]
        )

        source.add_narrow_band_image_lbdaobs(
            cube_ori,
            f"ORI_CORR_{num_line}",
            lbda=lbda_ori,
            width=nb_fwhm * fwhm_ori,
            method="max",
            subtract_off=False,
        )

        # Compute the spectra weighted by the correlation map for the
        # current line
        tags = [f"ORI_CORR_{num_line}"]
        source.extract_spectra(
            data_cube,
            obj_mask="ORI_MASK_OBJ",
            sky_mask="ORI_MASK_SKY",
            skysub=True,
            tags_to_try=tags,
        )
        source.extract_spectra(
            data_cube,
            obj_mask="ORI_MASK_OBJ",
            sky_mask="ORI_MASK_SKY",
            skysub=False,
            tags_to_try=tags,
        )

    # set REFSPEC to the spectrum weighted by the correlation map of the
    # brightest line
    num_max = source.lines["NUM_LINE"][np.argmax(source.lines["FLUX"])]
    source.header["REFSPEC"] = f"ORI_CORR_{num_max}_SKYSUB"

    hdulist.close()

    nb_par = Table(
        names=["LINE", "LBDA", "WIDTH", "MARGIN", "FBAND"],
        dtype=["U20", float, float, float, float],
        rows=nb_par_rows,
    )
    source.add_table(nb_par, "NB_PAR", select_in=None, col_dist=None)

    if save_to is not None:
        source.write(save_to)
    else:
        return source


def create_all_sources(
    cat3_sources,
    cat3_lines,
    origin_params,
    cube_cor_filename,
    cube_std_filename,
    mask_filename_tpl,
    skymask_filename_tpl,
    spectra_fits_filename,
    segmaps,
    version,
    profile_fwhm,
    out_tpl,
    *,
    n_jobs=1,
    author="",
    nb_fwhm=2,
    expmap_filename=None,
):
    """Create and save a MPDAF source file for each source.

    Parameters
    ----------
    cat3_sources: astropy.table.Table
        Table of unique sources (ORIGIN “Cat3_sources”).
    cat3_lines: astropy.table.Table
        Table of all the lines (ORIGIN “Cat3_lines”).
    origin_params: dict
        Dictionary of the parameters for the ORIGIN run.
    cube_cor_filename: str
        Name of the file containing the correlation cube of the ORIGIN run.
    cube_std_filename: str
        Name of the file containing the std cube of the ORIGIN run.
    mask_filename_tpl: str
        Template for the filename of the FITS file containing the mask of
        a source. The template is formatted with the id of the source.
        Eg: masks/source-mask-%0.5d.fits.
    skymask_filename_tpl: str:
        Template for the filename of the FITS file containing the mask of
        a sky for each source. The template is formatted with the id of the
        source. Eg: masks/sky-mask-%0.5d.fits.
    spectra_fits_filename: str
        Name of the FITS file containing the spectra of the lines.
    segmaps: dict(str: str)
        Dictionnary associating to a segmap type the associated FITS file name.
    version: str
        Version number stored in the source.
    profile_fwhm: list of int
        List of line profile FWHM in pixel. The index in the list is the
        profile number.
    out_tpl: str
        Template for the source file names. Eg. sources/source-%0.5d.fits
    author: str
        Name of the author.
    n_jobs: int
        Number of parallel processes used to create the source files.
    nb_fwhm: float
        Factor multiplying the FWHM of the line to compute the width of the
        narrow band image.
    expmap_filename: str
        Name of the file containing the exposure map.  If not None, a cut-out
        of the exposure map will be added to the source file.

    """
    job_list = []

    # Timestamp of the source creation
    source_ts = datetime.now().isoformat()

    for source_id in cat3_sources["ID"]:
        source_lines = cat3_lines[cat3_lines["ID"] == source_id]

        job_list.append(
            delayed(create_source)(
                source_id=source_id,
                source_table=cat3_sources,
                source_lines=source_lines,
                origin_params=origin_params,
                cube_cor_filename=cube_cor_filename,
                cube_std_filename=cube_std_filename,
                mask_filename=mask_filename_tpl % source_id,
                skymask_filename=skymask_filename_tpl % source_id,
                spectra_fits_filename=spectra_fits_filename,
                segmaps=segmaps,
                version=version,
                source_ts=source_ts,
                profile_fwhm=profile_fwhm,
                author=author,
                nb_fwhm=nb_fwhm,
                expmap_filename=expmap_filename,
                save_to=out_tpl % source_id,
            )
        )

    if job_list:
        Parallel(n_jobs=n_jobs)(progressbar(job_list))
