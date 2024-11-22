import pylidc
import csv
import functools
import glob
import os
import typing
import pandas as pd
import numpy as np
import SimpleITK as sitk


from luna.paths import get_paths_config

XYZ = typing.NewType('XYZ', typing.Tuple[float, float, float])


class CandidateInfo:
    # metadata class for candidate nodule information
    # classes as preferabled than namedtuples due to typing properties
    is_nodule: bool
    diameter_mm: float
    series_uid: str
    center_xyz: XYZ
    has_annontation: bool
    is_mal: bool

    def __init__(
            self,
            is_nodule: bool,
            diameter_mm: float,
            series_uid: str,
            center_xyz: XYZ,
            has_annontation: bool = None,
            is_mal: bool = None,
    ) -> None:
        self.is_nodule = is_nodule
        self.diameter_mm = diameter_mm
        self.series_uid = series_uid
        self.center_xyz = center_xyz
        self.has_annontation = has_annontation
        self.is_mal = is_mal

    def __lt__(self, other):
        # comparison function
        # when comparing two nodule candidates
        # we want to sort them by the fact whether they're a nodule
        # and if they're both nodles, then by size
        if self.is_nodule == other.is_nodule:
            return self.diameter_mm < other.diameter_mm
        else:
            return self.is_nodule < other.is_nodule

    def __str__(self) -> str:
        retval = (
            f'>> CandidateInfo\n' +
            f'series_uid: {self.series_uid}\n' +
            f'is_nodule: {self.is_nodule}\n' +
            f'diameter_mm: {self.diameter_mm}\n' +
            f'center_xyz: {self.center_xyz}\n'
            f'has_annontation: {self.has_annontation}\n'
            f'is_mal: {self.is_mal}\n'
        )
        return retval


@functools.lru_cache(1, typed=True)
def get_candidate_info_dict(w_malignancy: bool = False, require_on_disk: bool = True):
    """
    This function will map it so that each item contains ALL notules found in this series.
    Returns a dictionary:
        Each key is a series UID, and the values is a list of all nodules extracted from it. 
    """

    paths = get_paths_config()

    if w_malignancy is False:
        candidateInfo_list = get_candidate_info_list(require_on_disk=require_on_disk, verbose=False)
    else:
        candidateInfo_list = get_candidate_w_malignancy_list(require_on_disk=require_on_disk, verbose=False)
    candidateInfo_dict: typing.Dict[str, typing.List[CandidateInfo]] = {}

    for candidate_info in candidateInfo_list:
        candidateInfo_dict.setdefault(candidate_info.series_uid,
                                      []).append(candidate_info)

    return candidateInfo_dict


@functools.lru_cache(1, typed=True)
def get_candidate_info_list(require_on_disk: bool = True, verbose: bool = False) -> typing.List[CandidateInfo]:
    """Get information on nodule candidates found on disk.

    Args:
        require_on_disk (bool, optional): Whether to check if files are actually stored on disk. Defaults to True.
        verbose (bool, optional): Print useful logs. Defaults to False.

    Returns:
        typing.List[CandidateInfo]: A list of nodule candidate information.
    """
    paths = get_paths_config()

    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    #log('Scanning files present on disk...', verbose=verbose)
    mhd_list = glob.glob(os.path.join(paths.SUBSETS_PATTERN, '*.mhd'))
    # list all series uid but drop the extension (".mhd")
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    #log(f'Items found present on disk {len(presentOnDisk_set)}', verbose=verbose)

    diameter_dict: typing.Dict[str, typing.Tuple[XYZ, float]] = {}
    with open(paths.ANNOTATIONS, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz: XYZ = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    #log('Checking candidates and matching with annotations...', verbose=verbose)
    candidateInfo_list = []
    with open(paths.CANDIDATES, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # check if this uid is present on disk
            if require_on_disk is True and series_uid not in presentOnDisk_set:
                continue

            isNodule: bool = bool(int(row[4]))
            candidateCenter_xyz: XYZ = tuple([float(x) for x in row[1:4]])
            candidateDiameter_mm = 0.0

            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    # divides the diameter by 2 to get radius, and then divides the radius by 2
                    # to require that the two nodule center points not be too far apart
                    # relative to the size of the nodule
                    # (this is a bounding box check, not a true distance check)
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfo(
                is_nodule=isNodule,
                diameter_mm=candidateDiameter_mm,
                series_uid=series_uid,
                center_xyz=candidateCenter_xyz,
            ))

    #log(f'Found {len(candidateInfo_list)} candidates! Returning result...', verbose=verbose)
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1, typed=True)
def get_candidate_w_malignancy_list(require_on_disk: bool = True, verbose: bool = False) -> typing.List[CandidateInfo]:
    """
    This function will map it so that each item contains nodules found in the LUNA dataset.
    For this function to work, annontations with malignancy should have been generated beforehand.

    Returns a dictionary:
        Each key is a series UID, and the values is a list of all nodules extracted from it. 
    """

    paths = get_paths_config()

    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    #log('Scanning files present on disk...', verbose=verbose)
    mhd_list = glob.glob(os.path.join(paths.SUBSETS_PATTERN, '*.mhd'))
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    #log(f'Reading annotations with malignancy from {paths.ANNOTATIONS_WITH_MALIGNANCY}...', verbose=verbose)
    candidateInfo_list: typing.List[CandidateInfo] = []
    with open(paths.ANNOTATIONS_WITH_MALIGNANCY, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]

            candidateInfo_list.append(
                CandidateInfo(
                    is_nodule=True,
                    has_annontation=True,
                    is_mal=isMal_bool,
                    diameter_mm=annotationDiameter_mm,
                    series_uid=series_uid,
                    center_xyz=annotationCenter_xyz
                )
            )
    #log('Cross checking all malignancy nodules will ALL candidate nodules...', verbose=verbose)
    with open(paths.CANDIDATES, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # check if this uid is present on disk
            if require_on_disk is True and series_uid not in presentOnDisk_set:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfo(
                        is_nodule=False,
                        has_annontation=False,
                        is_mal=False,
                        diameter_mm=0.0,
                        series_uid=series_uid,
                        center_xyz=candidateCenter_xyz,
                    )
                )

    print(f'Found {len(candidateInfo_list)} candidates! Returning result...')
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


def generate_annotations_with_malignancy(verbose: bool = False):
    """
    This function does some *magic* found in Deep Learning with pytorch book, and its code can be found at:
    https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/generate_annotations_with_malignancy.ipynb

    Parsing all annotations will take a while, so expect a longer running time.

    You usually only need to run this once, and its results will be stored to annotations-with-malignancy.csv
    The full paths to the resulting file is found at LunaPaths.ANNOTATIONS_WITH_MALIGNANCY.

    Args:
        verbose (bool, optional): Print useful logs. Defaults to False.
    """
    paths = get_paths_config()

    annotations = pd.read_csv(paths.ANNOTATIONS)

    malignancy_data = []
    missing = []
    spacing_dict = {}
    scans = {s.series_instance_uid: s for s in pylidc.query(pylidc.Scan).all()}
    suids = annotations.seriesuid.unique()

    #log('Reading CT data and parsing malignancy data. This might take a while...', verbose=verbose)
    for suid in suids:
        fn = glob.glob(os.path.join(paths.SUBSETS_PATTERN, '{}.mhd'.format(suid)))
        if len(fn) == 0 or '*' in fn[0]:
            missing.append(suid)
            continue
        fn = fn[0]
        x = sitk.ReadImage(fn)
        spacing_dict[suid] = x.GetSpacing()
        s = scans[suid]
        for ann_cluster in s.cluster_annotations():
            # this is our malignancy criteron described in Chapter 14
            is_malignant = len([a.malignancy for a in ann_cluster if a.malignancy >= 4]) >= 2
            centroid = np.mean([a.centroid for a in ann_cluster], 0)
            bbox = np.mean([a.bbox_matrix() for a in ann_cluster], 0).T
            coord = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in centroid[[1, 0, 2]]])
            bbox_low = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[0, [1, 0, 2]]])
            bbox_high = x.TransformIndexToPhysicalPoint([int(np.round(i)) for i in bbox[1, [1, 0, 2]]])
            malignancy_data.append(
                (suid, coord[0],
                 coord[1],
                 coord[2],
                 bbox_low[0],
                 bbox_low[1],
                 bbox_low[2],
                 bbox_high[0],
                 bbox_high[1],
                 bbox_high[2],
                 is_malignant, [a.malignancy for a in ann_cluster]))

    #log('And now we match the malignancy data to the annotations. This is a lot faster...', verbose=verbose)
    df_mal = pd.DataFrame(malignancy_data, columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'bboxLowX',
                          'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ', 'mal_bool', 'mal_details'])
    processed_annot = []
    annotations['mal_bool'] = float('nan')
    annotations['mal_details'] = [[] for _ in annotations.iterrows()]
    bbox_keys = ['bboxLowX', 'bboxLowY', 'bboxLowZ', 'bboxHighX', 'bboxHighY', 'bboxHighZ']
    for k in bbox_keys:
        annotations[k] = float('nan')
    for series_id in annotations.seriesuid.unique():
        # series_id = '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860'
        # c = candidates[candidates.seriesuid == series_id]
        a = annotations[annotations.seriesuid == series_id]
        m = df_mal[df_mal.seriesuid == series_id]
        if len(m) > 0:
            m_ctrs = m[['coordX', 'coordY', 'coordZ']].values
            a_ctrs = a[['coordX', 'coordY', 'coordZ']].values
            #print(m_ctrs.shape, a_ctrs.shape)
            matches = (np.linalg.norm(a_ctrs[:, None] - m_ctrs[None], ord=2, axis=-1) / a.diameter_mm.values[:, None] < 0.5)
            has_match = matches.max(-1)
            match_idx = matches.argmax(-1)[has_match]
            a_matched = a[has_match].copy()
            # c_matched['diameter_mm'] = a.diameter_mm.values[match_idx]
            a_matched['mal_bool'] = m.mal_bool.values[match_idx]
            a_matched['mal_details'] = m.mal_details.values[match_idx]
            for k in bbox_keys:
                a_matched[k] = m[k].values[match_idx]
            processed_annot.append(a_matched)
            processed_annot.append(a[~has_match])
        else:
            processed_annot.append(c)
    processed_annot = pd.concat(processed_annot)
    processed_annot.sort_values('mal_bool', ascending=False, inplace=True)
    processed_annot['len_mal_details'] = processed_annot.mal_details.apply(len)

    #log("Finally, we drop NAs (where we didn't find a match) and save it in the right place:", paths.ANNOTATIONS_WITH_MALIGNANCY, verbose=verbose)
    df_nona = processed_annot.dropna()
    df_nona.to_csv(paths.ANNOTATIONS_WITH_MALIGNANCY, index=False)
