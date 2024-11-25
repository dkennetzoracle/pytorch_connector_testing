import os
import sys
from time import perf_counter

from streaming import StreamingDataset, StreamingDataLoader
from tqdm import tqdm

import oci
from oci.object_storage import ObjectStorageClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.parse_args import stream_from_bucket_parser
from utils.silly_utils import count_lists_in_key

def main():
    args = stream_from_bucket_parser()
    config = oci.config.from_file(file_location=args.oci_config_path,
                                  profile_name=args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    remote_bucket = f'oci://{args.bucket_name}@{namespace}/'
    dataset = StreamingDataset(local=args.local_cache,
                               remote=remote_bucket,
                               download_retry=3,
                               batch_size=64,
                               shuffle=True,
                               cache_limit=args.local_cache_max_size)
    data_loader = StreamingDataLoader(dataset, batch_size=64, num_workers=os.cpu_count())
    start = perf_counter()
    total_samples = 0
    log_file = args.log_file if args.log_file else "mds_log.txt"
    out = ""
    for epoch in range(2):
        num_samples = 0
        t0 = perf_counter()
        for data in tqdm(data_loader, smoothing=0, mininterval=1):
            num_samples += count_lists_in_key(data, 'text')
            if num_samples >= 1000000:
                break
        total_samples += num_samples
        out += f"For {__file__} on {epoch}, streamed over {num_samples} samples in {perf_counter() - t0}s or {num_samples / (perf_counter() - t0)} samples / sec."
    end = perf_counter()
    out += f"For {__file__}: streamed {total_samples} in {end - start}s or {total_samples / (end - start)} samples / sec."
    with open(log_file, 'w') as f:
        f.write(out)

if __name__ == "__main__":
    sys.exit(main())