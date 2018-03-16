"""Source map creation code."""
from astropy import units as u
from matplotlib import pyplot as plt
import numpy as np
from photutils import detect_sources, detect_threshold


def gen_source_mask(source_id, ra, dec, lines, detection_cube, threshold,
                    cont_sky, out_dir, *, mask_size=50, seg_npixel=5,
                    radec_is_xy=False, verbose=False):
    """Generate a mask for the source segmenting the detection cube.

    This function generates a mask for a source by combining the masks of each
    of its lines.  The mask of a line is created by segmenting the max image
    extracted from the detection cube around the line wavelength.  For primary
    ORIGIN sources, the correlation cube should be used, for complementary
    sources, the STD cube should be used.

    The sky mask of each source is computed by intersecting the reverse of the
    source mask and the continuum sky mask.
    TODO: Implement the sky mask.

    The function also check that all the border pixels of the mask are 0.  If
    that's not the case - that means that the mask size is to small or that
    there is a problem in the mask creation - the source identifier is returned
    else nothing is returned (i.e. None).

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
    out_dir: string
        Name of the output directory to create the masks in.
    mask_size: int
        Size in pixels of the (square) masks.
    seg_npixel:
        Minimum number of pixels used by photutils for the segmentation.
    radec_is_xy: bool
        True if the position is given in pixel instead of RA, Dec.
    verbose: true
        If true, the correlation map and the segmentation map images associated
        to each line will also be saved in the output directory.

    """
    # We will modify the table
    lines = lines.copy()

    if radec_is_xy:
        source_x, source_y = ra, dec
    else:
        source_x, source_y = detection_cube[0, :, :].wcs.wcs.all_world2pix(
            ra, dec, 0)

    sub_cube = detection_cube.subcube(
        center=(source_y, source_x), size=mask_size,
        unit_center=None, unit_size=None)

    sky_mask = cont_sky.subimage(
        center=(source_y, source_x), size=mask_size,
        unit_center=None, unit_size=None)

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
            wave=(min_z, max_z), unit_wave=None, agg_method="max")

        max_map.data[max_map.mask] = -9999.
        segmap = detect_sources(max_map.data, threshold, seg_npixel)

        # Segment associated to the line (maps are y, x)
        # The position to look for the value of the segment must be integers,
        # as we got the value by WCS computation, we round the value because we
        # may end with values with .999.
        x_line, y_line = np.round([x_line, y_line]).astype(int)
        seg_line = segmap.data[y_line, x_line]

        if verbose:
            max_map.write(f"{out_dir}/S{source_id}_L{num_line}_cor.fits")
            # Correlation map plot
            fig, ax = plt.subplots()
            im = ax.imshow(max_map.data, origin='lower')
            ax.scatter(x_line, y_line)
            fig.colorbar(im)
            fig.suptitle(f"S{source_id} / L{num_line} / correlation map")
            fig.savefig(f"{out_dir}/S{source_id}_L{num_line}_cor.png")
            plt.close(fig)
            # Segmap plot
            fig, ax = plt.subplots()
            im = ax.imshow(segmap.data, origin='lower')
            ax.scatter(x_line, y_line)
            fig.colorbar(im)
            fig.suptitle(f"S{source_id} / L{num_line} / seg {seg_line}")
            fig.savefig(f"{out_dir}/S{source_id}_L{num_line}_segmap.png")
            plt.close(fig)

        # Combine the line mask to the source mask with OR
        source_mask.data |= (segmap.data == seg_line)

    sky_mask[source_mask.data] = 0

    # If verbose, also plot the mask and sky mask of the source.
    if verbose:
        fig, ax = plt.subplots()
        im = ax.imshow(source_mask.data, origin='lower')
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

    # Convert the mask to integer before saving to FITS.
    source_mask.data = source_mask.data.astype(int)
    source_mask.write(f"{out_dir}/source-mask-%0.5d.fits" % source_id,
                      savemask="none")

    sky_mask.write(f"{out_dir}/sky-mask-%0.5d.fits" % source_id,
                   savemask="none")

    if is_wrong:
        return source_id
