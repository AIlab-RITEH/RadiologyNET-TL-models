import argparse
import sys

from torch.utils.data import DataLoader
from luna.candidate_info import get_candidate_info_list

# from .model import LunaModel
from luna.dataset import PrepcacheLunaDataset, Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset
from tqdm import tqdm


class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=1024,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--context-slices',
                            help='How many context slices to use when cropping areas of interest. Defaults to 1 (3-channel output crop)',
                            default=1,
                            type=int,
                            )
        # parser.add_argument('--scaled',
        #     help="Scale the CT chunks to square voxels.",
        #     default=False,
        #     action='store_true',
        # )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        print("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            PrepcacheLunaDataset(
                # sortby_str='series_uid',
                contextSlices_count=self.cli_args.context_slices,
                verbose=True
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )
        print('Begin caching over the dataloader...')
        for batch in tqdm(self.prep_dl):
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
