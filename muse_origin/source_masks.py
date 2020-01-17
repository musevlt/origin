"""Source map creation code."""
import logging

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from photutils import detect_sources

__all__ = ("gen_source_mask",)


def _touches_edge(array):
    """Return true if any of the border pixels is 1."""
    return np.any([array[0, :], array[-1, :], array[:, 0], array[:, -1]])


def _trimmed(array, border_size):
    """Return a view on the array in the border removed"""
    return array[border_size:-border_size, border_size:-border_size]


def _count_1(ma):
    """Count the number of pixels to 1 in the masked array"""
    return np.count_nonzero(ma[~ma.mask] == 1)


def _create_mask(
    source_id,
    ra,
    dec,
    lines,
    detection_cube,
    threshold,
    cont_sky,
    fwhm,
    out_dir,
    *,
    mask_size=25,
    seg_npixel=5,
    min_sky_pixels=100,
    fwhm_factor=2,
    verbose=False,
    unit_center=None,
    unit_size=None,
    step=1,
):
    """Create the initial source mask.

    Function to create the source mask. The resulting mask may be too large
    because the initial size may be extended when the source touches the edges
    of the mask or when there are not enough sky pixels.

    Return:
    - source mask
    - sky mask
    """
    logger = logging.getLogger(__name__)

    # FIXME: there is a bug if the mask size is even in pixels. For now, we
    # force the use of an odd size. If the size is angular, we first convert
    # to a square pixel size using the first step of the cube.
    if unit_size is not None:
        step = detection_cube.wcs.get_step(unit=unit_size)[0]
        mask_size = np.ceil(mask_size / step)
        unit_size = None
    if mask_size % 2 == 0:
        msg = "Mask size must be odd. Changing mask_size from %s to %s."
        logger.debug(msg, mask_size, mask_size + 1)
        mask_size += 1

    # We will modify the table
    lines = lines.copy()

    # Flag to keep track if something went wrong
    is_wrong = False

    sub_cube = detection_cube.subcube(
        center=(dec, ra), size=mask_size, unit_center=unit_center, unit_size=unit_size
    )
    # Take a copy of the subimage because the method returns a view on the map
    # and we are modifying it.
    sky_mask = cont_sky.subimage(
        center=(dec, ra), size=mask_size, unit_center=unit_center, unit_size=unit_size
    ).copy()

    # When the source is at the edge of the cube, the sky mask may be “masked”
    # for regions outside of the cube.  Ensure that they will be set to 0 (non
    # sky) when saving.
    sky_mask.data.fill_value = 0

    # Empty (0) source mask
    source_mask = sub_cube[0, :, :] * 0
    source_mask.data.fill_value = 0
    source_mask.data = source_mask.data.astype(bool)

    # Pixel position of the lines in the sub-cube
    lines["ra"].unit, lines["dec"].unit = u.deg, u.deg
    lines_x, lines_y = sub_cube.wcs.wcs.all_world2pix(lines["ra"], lines["dec"], 0)

    for x_line, y_line, z_line, fwhm_line, num_line in zip(
        lines_x, lines_y, lines["z"], lines["fwhm"], lines["num_line"]
    ):

        # Limit for the correlation map extraction, need to be integers.
        min_z = int(z_line - fwhm_line)
        max_z = int(z_line + fwhm_line)

        max_map = sub_cube.get_image(wave=(min_z, max_z), unit_wave=None, method="max")

        if max_map.mask is False or max_map.mask.shape == ():
            segmap = detect_sources(max_map.data, threshold, seg_npixel)
        else:
            segmap = detect_sources(
                max_map.data, threshold, seg_npixel, mask=max_map.mask
            )

        # Segment associated to the line (maps are y, x)
        # The position to look for the value of the segment must be integers,
        # as we got the value by WCS computation, we round the value because we
        # may end with values with .999.
        x_line, y_line = np.round([x_line, y_line]).astype(int)

        try:
            seg_line = segmap.data[y_line, x_line]
        except AttributeError:
            # photutils 0.7+ returns None when no sources are detected
            # (-> AttributeError)
            seg_line = 0
        except IndexError:
            # The line position is outside of the mask coverage. We must try
            # with a larger mask.
            is_wrong = True
            msg = (
                "The line %d associated to source %d is too far from the source "
                "position given the mask size (%d). The created mask is wrong."
            )
            logger.error(msg, num_line, source_id, mask_size)
            break  # Stop processing of this line.

        if seg_line != 0:
            line_mask = segmap.data == seg_line
        else:
            # If the segment of the line is 0, that means that there are not
            # enough pixels above the threshold to create a source with
            # photutils. The mask will consist only in the PSF added next.
            line_mask = np.full(max_map.shape, False, dtype=bool)

        # Adding the FWHM disk around the line position
        radius = int(np.ceil(0.5 * fwhm_factor * fwhm[z_line]))
        yy, xx = np.mgrid[: line_mask.shape[0], : line_mask.shape[1]]
        line_mask[((xx - x_line) ** 2 + (yy - y_line) ** 2) <= radius ** 2] = True

        if verbose:
            max_map.write(f"{out_dir}/S{source_id}_L{num_line}_step{step}_cor.fits")
            # Correlation map plot
            fig, ax = plt.subplots()
            im = ax.imshow(max_map.data, origin="lower")
            ax.scatter(x_line, y_line)
            fig.colorbar(im)
            fig.suptitle(f"S{source_id} / L{num_line} / correlation map")
            fig.savefig(f"{out_dir}/S{source_id}_L{num_line}_step{step}_cor.png")
            plt.close(fig)
            # Segmap plot
            fig, ax = plt.subplots()
            im = ax.imshow(segmap.data, origin="lower")
            ax.scatter(x_line, y_line)
            fig.colorbar(im)
            fig.suptitle(f"S{source_id} / L{num_line} / seg {seg_line}")
            fig.savefig(f"{out_dir}/S{source_id}_L{num_line}_step{step}_segmap.png")
            plt.close(fig)
            # Line mask plot
            fig, ax = plt.subplots()
            im = ax.imshow(line_mask, origin="lower")
            ax.scatter(x_line, y_line)
            fig.suptitle(f"S{source_id} / L{num_line} / mask")
            fig.savefig(f"{out_dir}/S{source_id}_L{num_line}_step{step}_mask.png")
            plt.close(fig)

        # Combine the line mask to the source mask with OR
        source_mask.data |= line_mask

    sky_mask[source_mask.data] = 0

    # If verbose, also plot the mask and sky mask of the source.
    if verbose:
        fig, ax = plt.subplots()
        im = ax.imshow(source_mask.data.astype(int), origin="lower")
        fig.colorbar(im)
        fig.suptitle(f"S{source_id} mask")
        fig.savefig(f"{out_dir}/S{source_id}_mask.png")
        im = ax.imshow(sky_mask.data, origin="lower")
        fig.suptitle(f"S{source_id} sky mask")
        fig.savefig(f"{out_dir}/S{source_id}_skymask.png")
        plt.close(fig)

    # OR combination because the mask may be wrong because of lines outside of
    # the mask.
    is_wrong |= (
        _touches_edge(source_mask.data) or _count_1(sky_mask.data) < min_sky_pixels
    )

    if is_wrong and step <= 4:
        new_size = int(mask_size * 1.5)
        logger.debug(
            "Source %s mask can't be done with size %s px at "
            "step %s. Trying with %s px.",
            source_id,
            mask_size,
            step,
            new_size,
        )
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
            min_sky_pixels=min_sky_pixels,
            fwhm_factor=fwhm_factor,
            verbose=verbose,
            unit_center=unit_center,
            unit_size=unit_size,
            step=step + 1,
        )

    if is_wrong:
        logger.error(
            "Source %s mask couldn't be done after %s attempts "
            "with a mask size up to %s.",
            source_id,
            step,
            mask_size,
        )

    return source_mask, sky_mask


def _trim_masks(source_mask, sky_mask, min_size, min_sky_npixels):
    """Trim the source and sky masks

    This function trims the source and sky masks to the minimum size ensuring
    that:
    - the size is at least `min_size`
    - the source does not touch the mask edges
    - the number of sky pixels is at least `min_sky_npixels`

    Returns:
    - source_mask
    - sky_mask
    - touch_edge: True if the source touches the edge of the mask
    - not_enough_sky: True is the are fewer sky pixels than `min_sky_npixels`
    """
    initial_size = len(source_mask.data)
    border_size = 1

    while (
        initial_size - 2 * border_size >= min_size
        and not _touches_edge(_trimmed(source_mask.data, border_size))
        and _count_1(_trimmed(sky_mask.data, border_size)) >= min_sky_npixels
    ):
        border_size += 1

    # Last border size before there is a problem.
    border_size -= 1

    if border_size > 1:
        source_mask = source_mask[border_size:-border_size, border_size:-border_size]
        sky_mask = sky_mask[border_size:-border_size, border_size:-border_size]
    touch_edge = _touches_edge(source_mask.data)
    not_enough_sky = _count_1(sky_mask.data) < min_sky_npixels

    return source_mask, sky_mask, touch_edge, not_enough_sky


def gen_source_mask(
    source_id,
    ra,
    dec,
    lines,
    detection_cube,
    threshold,
    cont_sky,
    fwhm,
    out_dir,
    *,
    mask_size=25,
    seg_npixel=5,
    min_sky_npixels=100,
    fwhm_factor=2,
    verbose=False,
    unit_center=None,
    unit_size=None,
):
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

    The size of the returned mask may be larger than the requested `mask_size`
    so that:
    - the source mask does not touch the edge of the mask
    - the number of sky pixels is at least `min_sky_npixels`

    If that's not possible, `source_id` is returned, else nothing is returned
    (i.e. None).

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
        Minimal size in pixels of the (square) masks.
    seg_npixel: int
        Minimum number of pixels used by photutils for the segmentation.
    min_sky_npixels: int
        Minimum number of sky pixels in the mask.
    fwhm_factor: float
        Factor applied to the FWHM when adding a disk at the line position.
    verbose: true
        If true, the correlation map and the segmentation map images associated
        to each line will also be saved in the output directory.

    """
    logger = logging.getLogger(__name__)

    source_mask, sky_mask = _create_mask(
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
        unit_size=unit_size,
    )

    source_mask, sky_mask, touch_edge, not_enough_sky = _trim_masks(
        source_mask, sky_mask, min_size=mask_size, min_sky_npixels=min_sky_npixels
    )

    if touch_edge:
        msg = "Mask creation problem: the source %s touches the edge of the mask."
        logger.error(msg, source_id)
    if not_enough_sky:
        msg = "Mask creation problem: the source %s has not enough sky pixels."
        logger.error(msg, source_id)

    # Convert the mask to integer before saving to FITS.
    source_mask.data = source_mask.data.astype(int)
    source_mask.write(f"{out_dir}/source-mask-%0.5d.fits" % source_id, savemask="none")

    sky_mask.write(f"{out_dir}/sky-mask-%0.5d.fits" % source_id, savemask="none")

    if touch_edge or not_enough_sky:
        return source_id
