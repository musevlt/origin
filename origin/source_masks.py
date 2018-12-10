"""Source map creation code."""
import logging

from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np
from photutils import detect_sources


def _create_mask(source_id, ra, dec, lines, detection_cube, threshold,
                 cont_sky, fwhm, out_dir, *, mask_size=50, seg_npixel=5,
                 fwhm_factor=2, verbose=False, unit_center=None,
                 unit_size=None, step=1):
    """
    Function to create the source mask. The resulting mask may be too large
    because the initial size may be extended when the source touches the edges
    of the mask.

    Return:
    - source mask
    - sky mask
    - is_wrong: True if there was a problem creating the mask
    """
    logger = logging.getLogger(__name__)

    # We will modify the table
    lines = lines.copy()

    sub_cube = detection_cube.subcube(
        center=(dec, ra), size=mask_size,
        unit_center=unit_center, unit_size=unit_size)
    # Take a copy of the subimage because the method returns a view on the map
    # and we are modifying it.
    sky_mask = cont_sky.subimage(
        center=(dec, ra), size=mask_size,
        unit_center=unit_center, unit_size=unit_size).copy()

    # When the source is at the edge of the cube, the sky mask may be “masked”
    # for regions outside of the cube.  We set these regions to 0 (not sky) in
    # the sky mask as we don't know their content.
    sky_mask[sky_mask.mask] = 0

    # Empty (0) source mask
    source_mask = sub_cube[0, :, :]
    source_mask.mask = np.zeros_like(source_mask.data)
    source_mask.data = np.zeros_like(source_mask.data, dtype=bool)

    # Pixel position of the lines in the sub-cube
    lines['ra'].unit, lines['dec'].unit = u.deg, u.deg
    lines_x, lines_y = sub_cube.wcs.wcs.all_world2pix(
        lines['ra'], lines['dec'], 0)

    for x_line, y_line, z_line, fwhm_line, num_line in zip(
            lines_x, lines_y, lines['z'], lines['fwhm'], lines['num_line']):

        # Limit for the correlation map extraction, need to be integers.
        min_z = int(z_line - fwhm_line)
        max_z = int(z_line + fwhm_line)

        max_map = sub_cube.get_image(
            wave=(min_z, max_z), unit_wave=None, method="max")

        segmap = detect_sources(max_map.data, threshold, seg_npixel,
                                mask=max_map.mask)

        # Segment associated to the line (maps are y, x)
        # The position to look for the value of the segment must be integers,
        # as we got the value by WCS computation, we round the value because we
        # may end with values with .999.
        x_line, y_line = np.round([x_line, y_line]).astype(int)
        seg_line = segmap.data[y_line, x_line]
        if seg_line != 0:
            line_mask = segmap.data == seg_line
        else:
            # If the segment of the line is 0, that means that there are not
            # enough pixels above the threshold to create a source with
            # photutils. The mask will consist only in the PSF added next.
            line_mask = np.full_like(segmap.data, False, dtype=bool)

        # Adding the FWHM disk around the line position
        radius = int(np.ceil(0.5 * fwhm_factor * fwhm[z_line]))
        yy, xx = np.mgrid[:line_mask.shape[0], :line_mask.shape[1]]
        line_mask[((xx - x_line)**2 + (yy - y_line)**2) <= radius**2] = True

        if verbose:
            max_map.write(
                f"{out_dir}/S{source_id}_L{num_line}step{step}_cor.fits")
            # Correlation map plot
            fig, ax = plt.subplots()
            im = ax.imshow(max_map.data, origin='lower')
            ax.scatter(x_line, y_line)
            fig.colorbar(im)
            fig.suptitle(f"S{source_id} / L{num_line} / correlation map")
            fig.savefig(
                f"{out_dir}/S{source_id}_L{num_line}step{step}_cor.png")
            plt.close(fig)
            # Segmap plot
            fig, ax = plt.subplots()
            im = ax.imshow(segmap.data, origin='lower')
            ax.scatter(x_line, y_line)
            fig.colorbar(im)
            fig.suptitle(f"S{source_id} / L{num_line} / seg {seg_line}")
            fig.savefig(
                f"{out_dir}/S{source_id}_L{num_line}step{step}_segmap.png")
            plt.close(fig)
            # Line mask plot
            fig, ax = plt.subplots()
            im = ax.imshow(line_mask, origin='lower')
            ax.scatter(x_line, y_line)
            fig.suptitle(f"S{source_id} / L{num_line} / mask")
            fig.savefig(
                f"{out_dir}/S{source_id}_L{num_line}step{step}_mask.png")
            plt.close(fig)

        # Combine the line mask to the source mask with OR
        source_mask.data |= line_mask

    sky_mask[source_mask.data] = 0

    # If verbose, also plot the mask and sky mask of the source.
    if verbose:
        fig, ax = plt.subplots()
        im = ax.imshow(source_mask.data.astype(int), origin='lower')
        fig.colorbar(im)
        fig.suptitle(f"S{source_id} mask")
        fig.savefig(f"{out_dir}/S{source_id}_mask.png")
        im = ax.imshow(sky_mask.data, origin='lower')
        fig.suptitle(f"S{source_id} sky mask")
        fig.savefig(f"{out_dir}/S{source_id}_skymask.png")
        plt.close(fig)

    # Count number of true pixels in the edges of the mask.  If it's not
    # 0 that means that the mask touch the edge for the image and there may
    # be a problem.
    is_wrong = (np.sum(source_mask.data[0, :]) +
                np.sum(source_mask.data[-1, :]) +
                np.sum(source_mask.data[:, 0]) +
                np.sum(source_mask.data[:, -1]))

    if is_wrong and step <= 4:
        new_size = int(mask_size * 1.5)
        logger.warning("Source %s mask can't be done with size %s px at "
                       "step %s. Trying with %s px.", source_id, mask_size,
                       step, new_size)
        return _create_mask(
            source_id=source_id,
            ra=ra,
            dec=dec,
            lines=lines,
            detection_cube=detection_cube,
            threshold=threshold,
            cont_sky=cont_sky,
            fwhm=fwhm,
            out_dir=out_dir,
            mask_size=new_size,
            seg_npixel=seg_npixel,
            fwhm_factor=fwhm_factor,
            verbose=verbose,
            unit_center=unit_center,
            unit_size=unit_size,
            step=step + 1)

    if is_wrong:
        logger.error("Source %s mask couldn't be done after %s attempts "
                     "with a mask size up to %s.", source_id, step, mask_size)

    return source_mask, sky_mask, is_wrong


def gen_source_mask(source_id, ra, dec, lines, detection_cube, threshold,
                    cont_sky, fwhm, out_dir, *, mask_size=50, seg_npixel=5,
                    fwhm_factor=2, verbose=False, unit_center=None,
                    unit_size=None):
    """Generate a mask for the source segmenting the detection cube.

    This function generates a mask for a source by combining the masks of each
    of its lines.  The mask of a line is created by segmenting the max image
    extracted from the detection cube around the line wavelength.  For primary
    ORIGIN sources, the correlation cube should be used, for complementary
    sources, the STD cube should be used.

    For each line, a disk with a diameter of the FWHM at the line position,
    multiplied by `fwhm_factor`, is added to the mask.

    The sky mask of each source is computed by intersecting the reverse of the
    source mask and the continuum sky mask.

    After creating the mask, this function checks that all the border pixels
    are 0. If that's not the case, that may mean that the mask size is too
    small. Several larger size of mask are then tried to fix the problem.

    When the mask is correctly created, nothing is returned (i.e. None).

    Parameters
    ----------
    source_id: str or int
        Source identifier used in file names.
    ra, dec: float
        Right Ascension and declination of the source (or pixel position if
        radec_is_xy is true).  The mask will be centred on this position.
    lines: astropy.table.Table
        Table containing all the lines associated to the source.  This table
        must contain these columns:
        - num_line: the identifier of the line (for verbose output)
        - ra, dec: the position of the line in degrees
        - z: the pixel position of the line in the wavelength axis
        - fwhm: the full with at half maximum of the line in pixels
    detection_cube: mpdaf.obj.Cube
        Cube the lines where detected in.
    threshold: float
        Threshold used for segmentation. Should be lower (e.g. 50%) than the
        threshold used for source detection.
    cont_sky: mpdaf.obj.Image
        Sky mask obtained from a segmentation of the continuum. Must be on the
        same spatial WCS as the detection cube. The pixels must have a value of
        1 at the sky positions, 0 otherwise.
    fwhm: numpy array of floats
        Value of the spatial FWHM in pixels at each wavelength of the detection
        cube.
    out_dir: string
        Name of the output directory to create the masks in.
    mask_size: int
        Size in pixels of the (square) masks.
    seg_npixel:
        Minimum number of pixels used by photutils for the segmentation.
    fwhm_factor: float
        Factor applied to the FWHM when adding a disk at the line position.
    verbose: true
        If true, the correlation map and the segmentation map images associated
        to each line will also be saved in the output directory.

    """
    #logger = logging.getLogger(__name__)

    source_mask, sky_mask, is_wrong = _create_mask(
        source_id=source_id,
        ra=ra,
        dec=dec,
        lines=lines,
        detection_cube=detection_cube,
        threshold=threshold,
        cont_sky=cont_sky,
        fwhm=fwhm,
        out_dir=out_dir,
        mask_size=mask_size,
        seg_npixel=seg_npixel,
        fwhm_factor=fwhm_factor,
        verbose=verbose,
        unit_center=unit_center,
        unit_size=unit_size)

    # Convert the mask to integer before saving to FITS.
    source_mask.data = source_mask.data.astype(int)
    source_mask.write(f"{out_dir}/source-mask-%0.5d.fits" % source_id,
                      savemask="none")

    sky_mask.write(f"{out_dir}/sky-mask-%0.5d.fits" % source_id,
                   savemask="none")

    if is_wrong:
        return source_id
