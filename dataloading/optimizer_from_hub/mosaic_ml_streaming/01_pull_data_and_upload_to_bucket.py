#!/usr/bin/env python3

import os
import sys

from datasets import load_dataset
import oci
from oci.object_storage import ObjectStorageClient
from streaming import MDSWriter

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.oci_utils import create_bucket_if_not_exists 
from utils.parse_args import hf_optimizer_parse_args

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def main():
    ## Perform format conversion for dataset.
    args = hf_optimizer_parse_args()
    data = load_dataset(
        args.dataset_name,
        name=args.dataset_subname,
        trust_remote_code=True,
        split="train",
        streaming=True
    )

    dtype_mapping = {"string": "str"}
    columns = {}
    for feature_name, feature in data.features.items():
        columns[feature_name] = dtype_mapping[feature.dtype]

    config = oci.config.from_file(file_location=args.oci_config_path,
                                  profile_name=args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    create_bucket_if_not_exists(object_storage_client=object_storage_client,
                                bucket_name=args.output_bucket,
                                namespace=namespace,
                                oci_compartment_id=args.compartment_id)
    remote_bucket = f'oci://{args.output_bucket}@{namespace}/'
    compression = 'zstd' if args.compress_data else None
    with MDSWriter(out=remote_bucket, columns=columns,
                   compression=compression,
                   max_workers=args.max_workers,
                   ) as writer:
        for item in data:
            writer.write(item)

    print(f"Successfully wrote mds output to {remote_bucket}")

    

if __name__ == "__main__":
    sys.exit(main())
    