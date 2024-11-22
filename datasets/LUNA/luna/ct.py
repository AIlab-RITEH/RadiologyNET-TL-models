import functools
import glob
import SimpleITK as sitk
import numpy as np
import os
import typing

import SimpleITK as sitk
import numpy as np

from luna.utils import xyz2irc, XyzTuple, IrcTuple
from luna.paths import get_paths_config
from luna.candidate_info import get_candidate_info_dict, CandidateInfo
from luna.disk import get_cache


raw_cache = get_cache('LUNA/ScriptsPostTrain/cache1')


@functools.lru_cache(1, typed=True)
def get_ct(series_uid: str):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def get_ct_sample_size(series_uid: str):
    ct = Ct(series_uid=series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Ct:
    hu_a: np.ndarray
    """The CT image."""

    series_uid: str
    """Series UID (identified) of this CT image."""

    origin_xyz: XyzTuple
    """
    XYZ is the patient coordinate system. This contains data on how to transform from patient
    coord system to array coordinate system IRC (IRC=index, row, column). 
    This is the origin point where patient coordinate system starts.
    This is read out from CT metadata.
    """
    vxSize_xyz: XyzTuple
    """
    Voxel size in patient coordinate system (XYZ)
    This is read out from CT metadata.
    """
    direction_a: np.ndarray
    """
    This is a 3x3 matrix used for switching between IRC and XYZ coord systems.
    This is read out from CT metadata.
    """

    def __init__(self, series_uid: str):
        paths = get_paths_config()

        mhd_path = glob.glob(
            os.path.join(paths.SUBSETS_PATTERN, f'{series_uid}.mhd')
        )[0]

        # read the CT image
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a)
        # clip the weird ranges, anything lower than -1000 will be mapped to -1000
        # any pixel value above 1000 will be mapped to 1000
        # anything in between remains the same.
        self.hu_a = ct_a

        self.series_uid = series_uid
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = get_candidate_info_dict(w_malignancy=True)[self.series_uid]

        # filter out all nodules, remove non-nodules
        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.is_nodule is True
        ]
        self.positive_mask = self.build_annotation_mask(self.positiveInfo_list)
        # First step (the sum() part) gives us a 1D vector (over the slices) with the number of voxels flagged in the mask in each slice
        # Second step (the nonzero() part) takes indices of the mask slices that have a nonzero count, which we make into a list
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2))
                                 .nonzero()[0].tolist())

    def build_annotation_mask(self, positiveInfo_list: typing.List[CandidateInfo], threshold_hu=-700):
        """
        From a list of positive candidates, find pixels which surround it.
        Each candidate is considered to be at the center of the nodule, so we start the search from there.
        We iterate over all the candidate centers and search outward, checking the 3D image for changes in pixel intensity
        (which corresponds to tissue density). If there is a significant change in  tissue density, we stop the search.
        The sensitivity of pixel intensity changes is given wih threshold_hu parameter.

        Args:
            positiveInfo_list (typing.List[CandidateInfo]): List of candidate nodule centers.
            threshold_hu (int, optional): Sensitivity of tissue density changes. Defaults to -700.

        Returns:
            np.ndarray: a segmentation/annonation mask of the image and the nodules.
                The nodule is marked with 1 (ones) and non-nodule tissue is marked with 0 (zeroes)
        """
        boundingBox_a = np.zeros_like(self.hu_a, dtype=bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def get_raw_candidate(self, center_xyz: XyzTuple, width_irc: IrcTuple):
        """_summary_

        Args:
            center_xyz (XyzTuple): This is the XYZ center of the candidate nodule.
            width_irc (IrcTuple): This is the desired size of the exported CT chunk which
                includes the candidate nodule.

        Returns:
            tuple:
            1) first element is the exported CT chunk, i.e., the CT slice centered around the nodule
            2) second element is the center IRC (the converted center coordinate)
        """
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list: typing.List[slice] = []
        # take note: center_irc always has 3 elements , those being (i, r, c)
        # so slice_list will also contain three elements, one for each axis
        # and each of these three elements will be a slice with a start and end index
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        # now that the slice lists have been calculated, convert the three-element array into a tuple
        # and use it to grab a CT slice from the original image
        slice_list_tuple: typing.Tuple[slice] = tuple(slice_list)
        ct_chunk = np.array(self.hu_a[slice_list_tuple])
        pos_chunk = np.array(self.positive_mask[slice_list_tuple])
        return ct_chunk, pos_chunk, center_irc
