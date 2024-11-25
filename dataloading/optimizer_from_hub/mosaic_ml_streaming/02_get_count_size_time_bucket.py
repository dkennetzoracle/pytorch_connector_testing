#!/usr/bin/env python3

import argparse
from datetime import datetime
import sys

import oci
from oci.object_storage import ObjectStorageClient

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate stats of the bucket data in step 1.")
    parser.add_argument("--compartment-id", required=True,
                        help="The compartment ID in OCI in which to create the bucket")
    parser.add_argument("--bucket-name", required=True,
                        help="The bucket in the compartment to evaluate.")
    return parser.parse_args()

def get_object_details(object_storage_client: ObjectStorageClient,
                       namespace: str,
                       bucket_name: str
):
    resp = object_storage_client.list_objects(
        namespace_name=namespace,
        bucket_name=bucket_name,
        fields='name,size,timeCreated'
    )
    sizes = []
    times = []
    for object in resp.data.objects:
        sizes.append(int(object.size) / (1024 * 1024))
        times.append(object.time_created)
    while resp.data.next_start_with:
        resp = resp = object_storage_client.list_objects(namespace,
                                                         bucket_name,
                                                         start=resp.data.next_start_with,
                                                         fields='name,size,timeCreated')
        for object in resp.data.objects:
            sizes.append(int(object.size) / (1024 * 1024))
            times.append(object.time_created)
    return sizes, sorted(times)
    
    

def main() -> int:
    args = parse_args()
    # Fixme - take non-default config and profile.
    config = oci.config.from_file()
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    sizes, times = get_object_details(object_storage_client=object_storage_client,
                                      namespace=namespace,
                                      bucket_name=args.bucket_name)
    print(times[-1] - times[0])
    print(f"{sum(sizes) / 1024}GB")

if __name__ == "__main__":
    sys.exit(main())