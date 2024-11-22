import os
import pickle
import functools


os.makedirs(os.path.join('/tmp', 'luna_cfg'), exist_ok=True)
LUNA_CONFIG_PATH = os.path.join('/tmp', 'luna_cfg', 'config.pkl')


class LunaPaths:
    """
    A Paths object containing config for LUNA dataset.
    """

    ROOT: str = ''
    """Root directory where LUNA dataset is downloaded and unzipped."""
    ANNOTATIONS: str = ''
    """Path to annotationss CSV.  The annotations.csv file contains information about some of the candidates that
    have been flagged as nodules.
    """
    ANNOTATIONS_WITH_MALIGNANCY: str = ''
    """Path to annotationss with malignancy preprocesssed CSV. This contains data from annotation.csv, but the data has been cleaned."""
    CANDIDATES: str = ''
    """Path to candidates CSV. The candidates.csv file contains information about all lumps that potentially look like
    nodules, whether those lumps are malignant, benign tumors, or something else altogether. """

    TMP_WORKDIR: str = ''
    """
    Folder for storing any inbetween temporary files related to LUNA dataset.
    This will NOT be created if it does not exist, so be sure to check its existence on your own!
    """

    SUBSETS_PATTERN: str = ''
    """
    Since there are multiple subsets in the LUNA dataset, this  is not a strict path.
    Insetad, it's a general pattern-like location of the subsets, could include shell-style wildcards.
    """

    def __init__(self, luna_root: str, save_config: bool = True):
        """Init a LUNA dataset config.

        Args:
            luna_root (str): The root folder where all LUNA-related files are found. These should be
                the root folder of *unzipped* LUNA data.
            save_config (bool, optional): After this config is initalised, it should be saved for later use.
                Then, all the other LUNA-related methods will just read the stored config and cache it.
                If False, nothing will be saved.
                Defaults to True.
        """

        self.ROOT = luna_root
        self.ANNOTATIONS = os.path.join(self.ROOT, 'annotations.csv')
        self.CANDIDATES = os.path.join(self.ROOT, 'candidates_V2.csv')
        self.TMP_WORKDIR = os.path.join(self.ROOT, '_RADIOLOGYNET_TMP')
        self.SUBSETS_PATTERN = os.path.join(self.ROOT, 'subset*')
        self.ANNOTATIONS_WITH_MALIGNANCY = os.path.join(self.TMP_WORKDIR, 'annotations-with-malignancy.csv')

        if save_config is True:
            self.save()

    def __str__(self):
        retval = [
            f'>> {"LUNA root dir:":20} {self.ROOT}',
            f'>> {"SUBSETS_PATTERN:":20} {self.SUBSETS_PATTERN}',
            f'>> {"ANNOTATIONS:":20} {self.ANNOTATIONS}',
            f'>> {"ANNOTATIONS_WITH_MALIGNANCY:":20} {self.ANNOTATIONS_WITH_MALIGNANCY}',
            f'>> {"CANDIDATES:":20} {self.CANDIDATES}',
            f'>> {"TMP_WORKDIR:":20} {self.TMP_WORKDIR}',
        ]
        return '\n'.join(retval)

    def __hash__(self) -> int:
        """
        Hash the string representation of this object. The hash will be used for functools cache because
        it requires objects to be hashable.
        """
        return hash(self.__str__())

    def save(self):
        """Save LUNA paths config to a tmp file."""

        filepath = LUNA_CONFIG_PATH

        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        print('Saved config to', filepath)


@functools.lru_cache(1)
def get_paths_config():
    filepath = LUNA_CONFIG_PATH
    if os.path.exists(filepath) is False:
        raise FileNotFoundError(f'No path config found at {filepath}.' +
                                ' Please make sure to init a config by constructing a LunaPaths object' +
                                ' and then saving it using the save() method!')

    with open(filepath, 'rb') as file:
        cfg: LunaPaths = pickle.load(file)
    return cfg
