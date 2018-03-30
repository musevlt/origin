"""Source file creation code."""
import logging
import os

from astropy import units as u
from astropy.table import Table
from joblib import Parallel, delayed
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect.source import Source
from numpy.ma import is_masked
from tqdm import tqdm as progressbar

from .version import __version__ as origin_version


def create_source(source_id, source_table, line_table, origin_params,
                  cube_cor_filename, mask_filename, skymask_filename,
                  spectra_fits_filename, version, profile_fwhm, *,
                  author="", nb_fwhm=2, size=5, expmap_filename=None,
                  fieldmap_filename=None, save_to=None):
    """Create a MPDAF source.

    This function create a MPDAF source object for the ORIGIN source.

    Parameters
    ----------
    source_id: int
        Identifier for the source in the source and line tables.
    source_table: astropy.table.Table
        Catalogue of sources like the Cat3_sources one.
    line_table: astropy.table.Table
        Catalogue of lines like the Cat3_lines one.
    origin_params: dict
        Dictionary of the parameters for the ORIGIN run.
    cube_cor_filename: str
        Name of the file containing the correlation cube of the ORIGIN run.
    mask_filename: str
        Name of the file containing the mask of the source.
    skymask_filename: str:
        Name of the file containing the sky mask of the source.
    spectra_fits_filename: str
        Name of the FITS file containing the spectra of the lines.
    version: str
        Version number stored in the source.
    profile_fwhm: list of int
        List of line profile FWHM in pixel. The index in the list is the
        profile number.
    author: str
        Name of the author.
    nb_fwhm: float
        Factor multiplying the FWHM of the line to compute the width of the
        narrow band image.
    size: float
        Side of the square used for cut-outs around the source position (for
        images and sub-cubes) in arc-seconds.
    expmap_filename: str
        Name of the file containing the exposure map.  If not None, a cut-out
        of the exposure map will be added to the source file.
    fieldmap_filename: str
        Name of the file containing the fieldmap.
    save_to: str
        If not None, the source will be saved to the given file.

    Returns
    -------
    mpdaf.sdetect.source.Source or None
        If save_to is used, the function returns None.

    """
    logger = logging.getLogger(__name__)

    # [0] is to get a Row not a table.
    source_info = source_table[source_table['ID'] == source_id][0]
    source_lines = line_table[line_table['ID'] == source_id]

    data_cube = Cube(origin_params['cubename']).subcube(
        (source_info['dec'], source_info['ra']), size=size, unit_size=u.arcsec
    )
    cube_correl = Cube(cube_cor_filename).subcube(
        (source_info['dec'], source_info['ra']), size=size, unit_size=u.arcsec
    )

    source = Source.from_data(
        source_info['ID'], source_info['ra'], source_info['dec'],
        ("ORIGIN", origin_version, os.path.basename(origin_params['cubename']),
         data_cube.primary_header.get('CUBE_V', ""))
    )

    # Information about the source in the headers
    source.header["SRC_V"] = version, "Source version"
    source.add_history("Source created with ORIGIN", author)

    source.header["OR_X"] = source_info['x'], "x position in pixels"
    source.header["OR_Y"] = source_info['y'], "y position in pixels"
    source.header["OR_SEG"] = (source_info['seg_label'],
                               "Label in the segmentation map")
    source.header["OR_V"] = origin_version, "ORIGIN version"

    # source_header_keyword: (key_in_origin_param, description)
    parameters_to_add = {
        "OR_PROF": ("profiles", "OR input, spectral profiles"),
        "OR_FSF": ("PSF", "OR input, FSF cube"),
        "OR_THL%02d": ("threshold_list", "OR input threshold per area"),
        "OR_NA": ("nbareas", "OR number of areas"),
        "preprocessing": {
            "OR_DCT": ("dct_order", "OR input, DCT order")},
        "areas": {
            "OR_PFAA": ("pfa", "OR input, PFA used to create the area map"),
            "OR_SIZA": ("maxsize", "OR input, maximum area size in pixels"),
            "OR_MSIZA": ("minsize", "OR input, minimum area size in pixels"),
        },
        "compute_PCA_threshold": {
            "OR_PFAT": ("pfa_test", "OR input, PFA test")},
        "compute_greedy_PCA": {
            "OR_FBG": ("Noise_population",
                       "OR input: fraction of spectra estimated"),
            "OR_ITMAX": ("itermax", "OR input, maximum number of iterations"),
        },
        "compute_TGLR": {
            "OR_NG": ("neighbors", "OR input, neighbors"),
            "OR_NS": ("NbSubcube", "OR input, nb of subcubes for spatial seg"),
        },
        "compute_purity_threshold": {
            "OR_DXY": ("tol_spat",
                       "OR input, spatial tolerance for merging (pix)"),
            "OR_DZ": ("tol_spec",
                      "OR input, spectral tolerance for merging (pix)"),
            "OR_SXY": ("spat_size",
                       "OR input, spatial size of the spatial filter"),
            "OR_SZ": ("spect_size",
                      "OR input, spectral length of the spect. filter"),
        },
        "compute_spectra": {
            "OR_NXZ": ("grid_dxy", "OR input, grid Nxy")},
    }

    def add_keyword(keyword, param, description, params):
        if param == "threshold_list" and param in params:
            for idx, threshold in enumerate(params['threshold_list']):
                source.header[keyword % idx] = (float("%0.2f" % threshold),
                                                description)
        elif param in params:
            source.header[keyword] = params[param], description
        else:
            logger.debug("Parameter %s absent of the parameter list.", param)

    for keyword, val in parameters_to_add.items():
        if isinstance(val, dict) and keyword in origin_params:
            for key, val2 in val.items():
                add_keyword(key, *val2, origin_params[keyword]['params'])
        else:
            add_keyword(keyword, *val, origin_params)

    source.header["COMP_CAT"] = source_info['comp']
    if source.COMP_CAT:
        threshold_keyword, purity_keyword = "threshold2", "purity2"
    else:
        threshold_keyword, purity_keyword = "threshold", "purity"
    source.header["OR_TH"] = (
        float("%0.2f" % origin_params[threshold_keyword]),
        "OR input, threshold")
    source.header["OR_PURI"] = (
        float("%0.2f" % origin_params[purity_keyword]),
        "OR input, purity")

    # Mini-cubes
    source.cubes["MUSE_CUBE"] = data_cube
    source.cubes["ORI_CORREL"] = cube_correl

    # Table of sources around the exported sources.
    y_radius, x_radius = size / data_cube.wcs.get_step(u.arcsec) / 2
    x_min, x_max = source_info['x'] - x_radius, source_info['x'] + x_radius
    y_min, y_max = source_info['y'] - y_radius, source_info['y'] + y_radius
    nearby_sources = (source_table['x'] >= x_min) & \
        (source_table['x'] <= x_max) & \
        (source_table['y'] >= y_min) & \
        (source_table['y'] <= y_max)
    source.tables['ORI_CAT'] = source_table['ID', 'ra', 'dec'][nearby_sources]

    # Maps
    # No need to use add_white_image or add_cube as we already have the
    # sub-cubes.
    source.images["MUSE_WHITE"] = data_cube.mean(axis=0)
    # The MAXMAP is the max of the correlation cube
    source.images["ORI_MAXMAP"] = cube_correl.max(axis=0)
    # Using add_image, the image size is taken from the white map.
    source.add_image(Image(mask_filename), "ORI_MASK_OBJ")
    source.add_image(Image(skymask_filename), "ORI_MASK_SKY")
    source.add_image(Image(origin_params['segmap']), "ORI_SEGMAP")
    if expmap_filename is not None:
        source.add_image(Image(expmap_filename), "EXPMAP")

    # Full source spectra
    source.extract_spectra(data_cube, obj_mask="ORI_MASK_OBJ",
                           sky_mask="ORI_MASK_SKY", skysub=True)
    source.extract_spectra(data_cube, obj_mask="ORI_MASK_OBJ",
                           sky_mask="ORI_MASK_SKY", skysub=False)
    source.spectra['ORI_CORR'] = (
        cube_correl * source.images['ORI_MASK_OBJ']).mean(axis=(1, 2))
    # Add the FSF information to the source and use this information to compute
    # the PSF weighted spectra.
    source.add_FSF(data_cube, fieldmap=fieldmap_filename)
    a, b, beta, _ = source.get_FSF()
    fwhm_fsf = b * data_cube.wave.coord() + a
    source.extract_spectra(data_cube, obj_mask="ORI_MASK_OBJ",
                           sky_mask="ORI_MASK_SKY", skysub=True, psf=fwhm_fsf,
                           beta=beta)
    source.extract_spectra(data_cube, obj_mask="ORI_MASK_OBJ",
                           sky_mask="ORI_MASK_SKY", skysub=False, psf=fwhm_fsf,
                           beta=beta)
    source.header["REFSPEC"] = "MUSE_PSF_SKYSUB"

    # Per line data: the line table, the spectrum of each line, the narrow band
    # map from the data and from the correlation cube.
    # Content of the line table in the source
    line_columns = ["NUM_LINE",
                    "RA",
                    "DEC",
                    "LBDA_OBS",
                    "FWHM",
                    "FLUX",
                    "GLR",
                    "PROF",
                    "PURITY"]
    line_units = [None,
                  u.deg,
                  u.deg,
                  u.Angstrom,
                  u.Angstrom,
                  u.erg / (u.s * u.cm**2),
                  None,
                  None,
                  None]
    line_fmt = ["d",
                ".2f",
                ".2f",
                ".2f",
                ".2f",
                ".1f",
                ".1f",
                "d",
                ".2f"]
    # If the line is a complementary one, the GLR column is replace by STD
    if source.COMP_CAT:
        line_columns[6] = "STD"

    # We put all the ORIGIN lines in an ORI_LINES tables but keep only the
    # unique lines in the LINES tables.
    source.add_table(source_lines, "ORI_LINES")

    # Table containing the information on the narrow band images.
    nb_par = Table(
        names=["LINE", "LBDA", "WIDTH", "MARGIN", "FBAND"],
        dtype=['U20', float, float, float, float]
    )

    for line in [_ for _ in source_lines if is_masked(_['merged_in'])]:
        num_line = line['num_line']
        ra_ori, dec_ori = line['ra'], line['dec']
        lbda_ori = line['lbda']
        prof = line['profile']
        fwhm_ori = (profile_fwhm[prof] *
                    data_cube.wave.get_step(unit=u.Angstrom))
        flux_ori = line['flux']
        if source.COMP_CAT:
            glr_std = line['STD']
        else:
            glr_std = line['T_GLR']
        purity = line['purity']

        source.add_line(
            cols=line_columns,
            values=[num_line, ra_ori, dec_ori, lbda_ori, fwhm_ori, flux_ori,
                    glr_std, prof, purity],
            units=line_units,
            fmt=line_fmt,
            desc=None)

        source.spectra[f"ORI_SPEC_{num_line}"] = Spectrum(
            filename=spectra_fits_filename,
            ext=(f"DATA{num_line}", f"STAT{num_line}")
        )

        source.add_narrow_band_image_lbdaobs(
            data_cube, f"NB_LINE_{num_line}", lbda=lbda_ori,
            width=nb_fwhm*fwhm_ori, is_sum=True, subtract_off=True,
            margin=10., fband=3.)
        nb_par.add_row([f"NB_LINE_{num_line}", lbda_ori, nb_fwhm * fwhm_ori,
                        10., 3.])

        # TODO: Do we want the sum or the max?
        source.add_narrow_band_image_lbdaobs(
            cube_correl, f"ORI_CORR_{num_line}", lbda=lbda_ori,
            width=nb_fwhm*fwhm_ori, is_sum=True, subtract_off=False)

    source.add_table(nb_par, "NB_PAR")

    if save_to is not None:
        source.write(save_to)
    else:
        return source


def create_all_sources(cat3_sources, cat3_lines, origin_params,
                       cube_cor_filename, mask_filename_tpl,
                       skymask_filename_tpl, spectra_fits_filename,
                       version, profile_fwhm, out_tpl, *,
                       n_jobs=1, author="", nb_fwhm=2, size=5,
                       expmap_filename=None, fieldmap_filename=None):
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
    size: float
        Side of the square used for cut-outs around the source position (for
        images and sub-cubes) in arc-seconds.
    expmap_filename: str
        Name of the file containing the exposure map.  If not None, a cut-out
        of the exposure map will be added to the source file.
    fieldmap_filename: str
        Name of the file containing the fieldmap.

    """
    job_list = []

    for source_id in cat3_sources['ID']:
        job_list.append(delayed(create_source)(
            source_id=source_id,
            source_table=cat3_sources,
            line_table=cat3_lines,
            origin_params=origin_params,
            cube_cor_filename=cube_cor_filename,
            mask_filename=mask_filename_tpl % source_id,
            skymask_filename=skymask_filename_tpl % source_id,
            spectra_fits_filename=spectra_fits_filename,
            version=version,
            profile_fwhm=profile_fwhm,
            author=author,
            nb_fwhm=nb_fwhm,
            size=size,
            expmap_filename=expmap_filename,
            fieldmap_filename=fieldmap_filename,
            save_to=out_tpl % source_id
        ))

    if job_list:
        Parallel(n_jobs=n_jobs)(progressbar(job_list))