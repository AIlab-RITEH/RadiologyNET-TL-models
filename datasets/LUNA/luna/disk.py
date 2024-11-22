import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import BytesType, MODE_BINARY, BytesIO


def get_cache(directory: str):
    return FanoutCache(
        directory,
        shards=64,
        timeout=1,
        size_limit=3e11,
        # disk_min_file_size=2**20,
    )
