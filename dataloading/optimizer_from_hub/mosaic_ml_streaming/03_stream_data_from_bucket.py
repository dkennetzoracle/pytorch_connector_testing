import os
import sys

from streaming import StreamingDataset, StreamingDataLoader, Stream

import oci
from oci.object_storage import ObjectStorageClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.parse_args import stream_from_bucket_parser

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
    data_loader = StreamingDataLoader(dataset, batch_size=64)
    for i, batch in enumerate(data_loader):
        print(batch)
        if i == 1:
            sys.exit()

if __name__ == "__main__":
    sys.exit(main())