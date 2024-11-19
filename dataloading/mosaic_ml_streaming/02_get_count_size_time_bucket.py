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
    list_object_response = object_storage_client.list_objects(
        namespace_name=namespace,
        bucket_name=bucket_name,
        fields='name,size,timeCreated'
    )
    print(list_object_response.data)
    objects = list_object_response.data.objects
    

def main() -> int:
    args = parse_args()
    # Fixme - take non-default config and profile.
    config = oci.config.from_file()
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    get_object_details(object_storage_client=object_storage_client,
                       namespace=namespace,
                       bucket_name=args.bucket_name)

if __name__ == "__main__":
    sys.exit(main())