import functools
import glob
import random
import SimpleITK as sitk
import numpy as np
import os
import typing
import copy

import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from luna.disk import get_cache
from luna.candidate_info import CandidateInfo, get_candidate_info_dict, get_candidate_info_list, get_candidate_w_malignancy_list
from luna.paths import get_paths_config, LunaPaths
from luna.ct import get_ct, get_ct_sample_size

from luna.utils import XyzTuple, IrcTuple


# Note !! this seems to be old code from
#  https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p2ch13/dsets.py
# this class is somewhat equivalent to MaskTuple. But in their code, MaskTuple isn't really used anywhere.
# class LunaMask:
#     raw_dense_mask = None
#     dense_mask = None
#     body_mask = None
#     air_mask = None
#     raw_candidate_mask = None
#     candidate_mask = None
#     lung_mask = None
#     neg_mask = None
#     pos_mask = None


raw_cache = get_cache('data_cache')


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(series_uid: str, center_xyz: XyzTuple, width_irc: IrcTuple):
    """
    Args:
        center_xyz (XyzTuple): This is the XYZ center of the candidate nodule.
        width_irc (IrcTuple): This is the desired size of the exported CT chunk which
            includes the candidate nodule.

    Returns:
        tuple:
        1) first element is the exported CT chunk, i.e., the CT slice centered around the nodule
        2) second element is the center IRC (the converted center coordinate)
    """
    ct = get_ct(series_uid)
    ct_chunk, mask_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, mask_chunk, center_irc


class Luna2dSegmentationDataset(Dataset):
    pos_list: typing.List[CandidateInfo]
    """List of positive nodules."""
    contextSlices_count: int
    """
    How many slices of context to take when retrieving image chunks.
    E.g. if this equals to 3, then the number of returned channels in the image chunk
    is 7 (the main slice + 3 slices behind it + 3 slice in front of it).
    If this number is 1, then there will be 3 channels in the image chunk.
    """
    paths: LunaPaths
    """Config object containing pathnames to important directories and files."""
    is_val_set: typing.Union[None, bool]
    """
    If None, then this is a general dataset.
    If True, then this is the validation dataset.
    If False, then this is the training dataset.
    """
    full_ct: bool
    """When full_ct is True, we will use every slice in the CT for our
    dataset. If this is False, then only parts of the CT with positive samples in them will be used."""

    def __init__(
        self,
        val_stride=0,
        is_val_set: bool = None,
        series_uids: typing.List[str] = None,
        contextSlices_count: int = 1,
        full_ct: bool = False,
        verbose: bool = False,
    ):
        """Init a torch LUNA dataset.

        Args:
            paths (LunaPaths): _description_
            val_stride (int, optional): Just about every project will need to separate samples into a training set and a validation set.
                We are going to do that here by designating every tenth sample, specified by
                the val_stride parameter, as a member of the validation set. Defaults to 0.
            is_val_set (bool, optional): We will also accept an
                is_val_set parameter and use it to determine whether we should keep only the
                training data, the validation data, or everything.. Defaults to None.
            series_uids (List[str], optional): If we pass in a truthy series_uid, then the instance will only have nodules from that
                series. We can pass multiple series UIDs is we want multiple specific series to be used.
                This can be useful for visualization or debugging. Defaults to None, meaning all series will be used.
            full_ct(bool, optional): when full_ct is True, we will use every slice in the CT for our
                dataset. If this is False, then only parts of the CT with positive samples in them will be used. Defaults to False.
            contextSlices_count (int, optional): How many slices of context to take when retrieving image chunks.
                E.g. if this equals to 3, then the number of returned channels in the image chunk
                is 7 (the main slice + 3 slices behind it + 3 slice in front of it).
                If this number is 1, then there will be 3 channels in the image chunk.
            verbose(bool, optional): print useful logs. Defaults to False.

        """
        self.paths = get_paths_config()

        self.contextSlices_count = contextSlices_count
        self.full_ct = full_ct
        self.is_val_set = is_val_set

        if series_uids:
            self.series_list = series_uids
        else:
            self.series_list = sorted(get_candidate_info_dict(w_malignancy=True).keys())

        if is_val_set:
            assert val_stride > 0, val_stride
            # if validation set, take the every val_stride-th element.
            # if val_stride is 3, then then every 3rd element will be kept
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            # is_val_set is False or None and val_stride is greater than 0
            # if training or general set, remove every val_stride-th element
            # if val_stride is 3, then then every 3rd element will be removed, and the elements
            # in between will be kept
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        print('Iterating over all series, finding samples...')
        __series_uid_iterator = tqdm(self.series_list) if verbose is True else self.series_list
        for series_uid in __series_uid_iterator:
            index_count, positive_indexes = get_ct_sample_size(series_uid)

            if self.full_ct:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_ndx)
                                     for slice_ndx in positive_indexes]

        self.candidateInfo_list = get_candidate_w_malignancy_list()

        series_set = set(self.series_list)  # use a "set" object for faster lookup
        # Filter out the candidates from series not in our set
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidateInfo_list if nt.is_nodule]

        #log(str(self), verbose=verbose)

    def __str__(self) -> str:
        return "{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[self.is_val_set],
            len(self.sample_list),
            len(self.pos_list),
        )

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx: typing.Union[int, slice]):
        if type(ndx) == slice:
            # this code is used for slicing, when a slice is provided as index (e.g. array[0:2])
            start = ndx.start if ndx.start is not None else 0
            stop = ndx.stop if ndx.stop is not None else len(self)
            step = ndx.step if ndx.step is not None else 1

            retval: typing.List[Luna2dSegmentationItem] = []
            for i in range(start, stop, step):
                series_uid, slice_ndx = self.sample_list[i % len(self.sample_list)]
                item = self.getitem_fullSlice(series_uid, slice_ndx)
                retval.append(item)
            return retval
        else:
            series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
            return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid: str, slice_ndx: int):
        ct = get_ct(series_uid)
        ct_t = torch.zeros((self.get_channel_num(), 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            # When we reach  beyond the bounds of
            # the ct_a, we duplicate the first or last slice.
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        retval = Luna2dSegmentationItem(
            ct=ct_t,
            pos=pos_t,
            series_uid=ct.series_uid,
            slice_index=slice_ndx
        ).to_dict()

        return retval

    def get_channel_num(self):
        """
        Get the number of output channels this dataset returns.
        This is equal to contextSlices_count * 2 + 1
        Meaning that:
            if there are 1 context slices, there will be 3 output channels.
            if there are 3 context slices, there will be 7 output channels.
        """
        return self.contextSlices_count * 2 + 1


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # set seed for random crop reproducibility
        random.seed(571993)

        self.ratio_int = 2

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __len__(self):
        # training is done only on positive samples and cropped nodules
        # so len() method is a bit different than the superclass.
        # return len(self.pos_list)
        return 300000

    def __getitem__(self, ndx: typing.Union[slice, int]):
        if type(ndx) == slice:
            # this code is used for slicing, when a slice is provided as index (e.g. array[0:2])
            start = ndx.start if ndx.start is not None else 0
            stop = ndx.stop if ndx.stop is not None else len(self)
            step = ndx.step if ndx.step is not None else 1

            retval: typing.List[Luna2dSegmentationItem] = []
            for i in range(start, stop, step):
                candidateInfo_tup = self.pos_list[i % len(self.pos_list)]
                item = self.getitem_trainingCrop(candidateInfo_tup)
                retval.append(item)
            return retval
        else:
            candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
            return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo: CandidateInfo):
        cropsize_irc: IrcTuple = IrcTuple(index=self.get_channel_num(), row=96, col=96)
        ct_a, pos_a, center_irc = get_ct_raw_candidate(
            series_uid=candidateInfo.series_uid,
            center_xyz=candidateInfo.center_xyz,
            width_irc=cropsize_irc,
        )
        # functools has difficulty keeping track of variable types
        # so typecast everything for easier reading
        # note that these are the same types get_ct_raw_candidate() returns
        ct_a: np.ndarray = ct_a
        pos_a: np.ndarray[bool] = pos_a
        center_irc: IrcTuple = center_irc

        # Taking a one-element slice keeps the third dimension, which will be the (single) output channel.
        # this will basically take the middle slice and remove the surrounding slices
        pos_a = pos_a[self.contextSlices_count: self.contextSlices_count+1]
        # With two random numbers between 0 and 31, we crop both CT and mask
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:,  # cropping across all channels
                                     row_offset:row_offset+64,
                                     col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:,  # cropping across all channels
                                       row_offset:row_offset+64,
                                       col_offset:col_offset+64]).to(torch.long)
        # at this point, ct_t is of size (channel_num, 64, 64)
        # and pos_t is of size (1, 64, 64).
        # channel_num is determined by contextSlices_count

        retval = Luna2dSegmentationItem(
            ct=ct_t,# ct?aa
            pos=pos_t,# ps?a
            series_uid=candidateInfo.series_uid,
            slice_index=center_irc.index
        ).to_dict()

        return retval


class Luna2dSegmentationItem:
    """
    This is a class containing attributes for Luna 2d Segmentation dataset.

    It should also support unpacking, e.g.:
    item: Luna2dSegmentationItem = Luna2dSegmentationDataset()[0]
    ct, pos, series_uid, slice_index = item  ## --> unpacking works!
    """

    ct: torch.Tensor
    """
    The CT image, i.e., this is a slice of the original CT image. A single CT image
    may have multiple slices, and this object contains one of them.
    """

    pos: torch.Tensor
    """The positive mask. Elements are Boolean. False where non-nodule, True where there is a nodule."""

    series_uid: str
    """Series UID this CT belongs to."""

    slice_index: int
    """This object contains information from a single slice from the original CT image.
    The attribute slice_index tells us which slice exactly is this. Using this attribute
    we can locate this exact slice in the original CT volume.
    """

    def __init__(self, ct: torch.Tensor, pos: torch.Tensor, series_uid: str, slice_index: int) -> None:
        self.ct = ct
        self.pos = pos
        self.series_uid = series_uid
        self.slice_index = slice_index

    def __str__(self) -> str:
        retval = [
            f'CT size: {self.ct.size()}',
            f'Positive Mask size: {self.pos.size()}',
            f'Slice index: {self.slice_index}',
            f'Series UID: {self.series_uid}',
        ]
        return ' --- '.join(retval)

    #################################
    # These functions are mostly used to be able to unpack an object from this class
    # for example:
    # item = Luna2dSegmentationDataset()[0]
    # ct, pos, series_uid, slice_index = item  ## --> unpacking works!

    def to_tuple(self):
        return tuple([self.ct, self.pos, self.series_uid, self.slice_index])

    def __len__(self):
        return len(self.to_tuple())

    def __iter__(self):
        return iter(self.to_tuple())

    def __getitem__(self, index: int):
        return self.to_tuple()[index]

    def to_dict(self):
        retval = {
            'ct': self.ct,
            'pos': self.pos,
            'series_uid': self.series_uid,
            'slice_index': self.slice_index,
        }
        return retval
    #################################

    def plot_ct_with_mask(self, ax):
        """using a matplotlib.Axes object, plot the CT image
        and overlay its mask.

        Args:
            ax (matplotlib.Axes): Axes object on which to plot the CT image and overlay.

        Returns:
            the axes on which the images were plotted.
        """
        detached_ct = self.ct.detach().numpy()
        detached_pos = self.pos.detach().numpy()
        # take the middle slice (there could be multiple slices depending on contextSlices_count)
        detached_ct = detached_ct[len(detached_ct) % 2]
        # take the first slice of the mask since its first dimension is always 1
        detached_pos = detached_pos[0]
        #ax.imshow(detached_ct, cmap='gray')
        ax.imshow(detached_pos, cmap='gray', alpha=1)
        return ax


class PrepcacheLunaDataset(Dataset):
    def __init__(self, contextSlices_count: int = 1, verbose: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.contextSlices_count = contextSlices_count
        self.candidateInfo_list = get_candidate_w_malignancy_list()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.is_nodule]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx: int):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo = self.candidateInfo_list[ndx]
        cropsize_irc: IrcTuple = IrcTuple(index=self.get_channel_num(), row=96, col=96)
        get_ct_raw_candidate(
            series_uid=candidateInfo.series_uid,
            center_xyz=candidateInfo.center_xyz,
            width_irc=cropsize_irc,
        )

        series_uid = candidateInfo.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            get_ct_sample_size(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1  # candidate_t, pos_t, series_uid, center_t

    def get_channel_num(self):
        """
        Get the number of output channels this dataset returns.
        This is equal to contextSlices_count * 2 + 1
        Meaning that:
            if there are 1 context slices, there will be 3 output channels.
            if there are 3 context slices, there will be 7 output channels.
        """
        return self.contextSlices_count * 2 + 1
