__all__ = ("split_source", "update_masks", "update_sources")

import numpy as np
import logging
from .lib_origin import create_masks
from .source_creation import create_source
from datetime import datetime


def merge_sources(
    source_id, source_idlist, source_table, source_lines,
):
    logger = logging.getLogger(__name__)
    # check that source_id exist in source_table
    if source_id not in source_table['ID']:
        logger.error('Source %d not found in source table', source_id)
        return False

    # select in lines the relevant lines
    ksel = np.in1d(source_lines['ID'], source_idlist)
    if np.sum(ksel) == 0:
        logger.error('No lines found for source %s in line table', source_idlist)
        return False

    # attach lines to the master source ID
    source_lines['ID'][ksel] = source_id

    # remove merged source from source table
    ksel = np.in1d(source_table['ID'], source_idlist, invert=True)
    source_table = source_table[ksel]

    # update source table
    update_source_table(source_id, source_table, source_lines)

    return True


def split_source(
    source_id, num_lines_to_keep, source_table, source_lines, create_new=True
):
    """Split an ORIGIN source into two.

    This function split an ORIGIN source into two sources. 
    Parameters
    ----------
    source_id : int
        Identifier for the source in the source and line tables.
    num_lines_to_keep : list of int
        List of lines num from source_lines to keep in this source
    source_table : astropy.table.Table
        Catalogue of sources like the Cat3_sources one.
    source_lines : astropy.table.Table
        Catalogue of lines like the Cat3_lines one.
    create_new : bool
        If True a new entry is added in the catalog and the splitted lines are moved to it.

    Returns
    -------
    new_id : int
        The ID of the new source
    
    The num_lines_to_keep are kept into the current source_id, while a new entry will be added to the catalog
    for the rest of the lines. The catalog source_table and source_lines are modified


    """
    logger = logging.getLogger(__name__)

    # perform checks
    lines = source_lines[source_lines['ID'] == source_id]
    if len(lines) < 2:
        logger.error(
            'Only %d lines found in source id %d, need at least 2',
            len(lines),
            source_id,
        )
        return
    for k in num_lines_to_keep:
        if k not in lines['num_line']:
            logger.error('lines id %d not found in source id %d', k, source_id)
            return

    # find the remaining lines for the new source
    new_lines = [k for k in lines['num_line'] if k not in num_lines_to_keep]

    # find the new_id
    if create_new:
        new_id = source_lines['ID'].max() + 1
        logger.debug('Create new source %d with %s lines', new_id, new_lines)
    else:
        logger.debug('Removing %s lines from the current source', new_lines)

    # update new_lines
    for num in new_lines:
        ksort = source_lines['num_line'] == num
        if create_new:
            source_lines['ID'][ksort] = new_id
        else:
            # the lines to remove are set to ID = -99
            source_lines['ID'][ksort] = -99

    # update source table
    update_source_table(source_id, source_table, source_lines)

    if create_new:

        # add a new entry in source_table
        group = source_lines[source_lines['ID'] == new_id]
        result = {'ID': new_id}
        result['ra'] = np.average(group['ra'], weights=group['flux'])
        result['dec'] = np.average(group['dec'], weights=group['flux'])

        result['x'] = np.average(group['x'], weights=group['flux'])
        result['y'] = np.average(group['y'], weights=group['flux'])

        # The number of lines in the source is the number of lines that have
        # not been merged in another one.
        result['n_lines'] = np.sum(group['merged_in'] == -9999)

        result['seg_label'] = group['seg_label'][0]
        result['comp'] = group['comp'][0]  # FIXME: not necessarily true
        result['line_merged_flag'] = np.any(group["line_merged_flag"])

        ngroup = group[group['merged_in'] == -9999]
        result['flux'] = np.max(ngroup['flux'])
        result['T_GLR'] = np.max(ngroup['T_GLR'])
        result['nsigTGLR'] = np.max(ngroup['nsigTGLR'])
        result['STD'] = np.max(ngroup['STD'])
        result['nsigSTD'] = np.max(ngroup['nsigSTD'])
        result['purity'] = np.max(ngroup['purity'])
        result['waves'] = ','.join([str(int(l)) for l in ngroup['lbda'][:-4:-1]])

        source_table.add_row(result)

    return new_id if create_new else None


def update_masks(
    source_idlist,
    line_table,
    source_table,
    profile_fwhm,
    cube_correl,
    threshold_correl,
    cube_std,
    threshold_std,
    segmap,
    fwhm,
    out_dir,
    *,
    mask_size=25,
    min_sky_npixels=100,
    seg_thres_factor=0.5,
    fwhm_factor=2,
    plot_problems=True,
):

    """Create the mask of a list of sources.
    """
    logger = logging.getLogger(__name__)

    ksel = np.in1d(source_table['ID'], source_idlist)
    sel_source_table = source_table[ksel]
    if len(sel_source_table) == 0:
        logger.error('ID %s not found in source_table', source_idlist)
        return
    ksel = np.in1d(line_table['ID'], source_idlist)
    sel_line_table = line_table[ksel]
    if len(sel_line_table) == 0:
        logger.error('ID %s not found in line_table', source_idlist)
        return

    create_masks(
        line_table=sel_line_table,
        source_table=sel_source_table,
        profile_fwhm=profile_fwhm,
        cube_correl=cube_correl,
        threshold_correl=threshold_correl,
        cube_std=cube_std,
        threshold_std=threshold_std,
        segmap=segmap,
        fwhm=fwhm,
        out_dir=out_dir,
        mask_size=mask_size,
        min_sky_npixels=min_sky_npixels,
        seg_thres_factor=seg_thres_factor,
        fwhm_factor=fwhm_factor,
        plot_problems=plot_problems,
    )


def update_sources(
    source_idlist,
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
    author="",
    nb_fwhm=2,
    expmap_filename=None,
):
    logger = logging.getLogger(__name__)
    # Timestamp of the source creation
    source_ts = datetime.now().isoformat()

    for source_id in source_idlist:

        logger.debug('Creating source %d', source_id)
        source_lines = cat3_lines[cat3_lines["ID"] == source_id]

        create_source(
            source_id,
            cat3_sources,
            source_lines,
            origin_params,
            cube_cor_filename,
            cube_std_filename,
            mask_filename_tpl % source_id,
            skymask_filename_tpl % source_id,
            spectra_fits_filename,
            segmaps,
            version,
            source_ts,
            profile_fwhm,
            author=author,
            nb_fwhm=nb_fwhm,
            expmap_filename=expmap_filename,
            save_to=out_tpl % source_id,
        )


def update_source_table(source_id, source_table, source_lines):

    # update source_table
    ksel = source_table['ID'] == source_id
    group = source_lines[source_lines['ID'] == source_id]

    source_table['ra'][ksel] = np.average(group['ra'], weights=group['flux'])
    source_table['dec'][ksel] = np.average(group['dec'], weights=group['flux'])

    source_table['x'][ksel] = np.average(group['x'], weights=group['flux'])
    source_table['y'][ksel] = np.average(group['y'], weights=group['flux'])

    # The number of lines in the source is the number of lines that have
    # not been merged in another one.
    source_table['n_lines'][ksel] = np.sum(group['merged_in'] == -9999)

    source_table['seg_label'][ksel] = group['seg_label'][0]
    source_table['comp'][ksel] = group['comp'][0]  # FIXME: not necessarily true
    source_table['line_merged_flag'][ksel] = np.any(group["line_merged_flag"])

    ngroup = group[group['merged_in'] == -9999]
    source_table['flux'][ksel] = np.max(ngroup['flux'])
    source_table['T_GLR'][ksel] = np.max(ngroup['T_GLR'])
    source_table['nsigTGLR'][ksel] = np.max(ngroup['nsigTGLR'])
    source_table['STD'][ksel] = np.max(ngroup['STD'])
    source_table['nsigSTD'][ksel] = np.max(ngroup['nsigSTD'])
    source_table['purity'][ksel] = np.max(ngroup['purity'])
    ngroup.sort('flux')
    source_table['waves'][ksel] = ','.join(
        [str(int(l)) for l in ngroup['lbda'][:-4:-1]]
    )
