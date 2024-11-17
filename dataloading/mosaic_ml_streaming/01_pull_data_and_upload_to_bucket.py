#!/usr/bin/env python3

import argparse
import os
import sys

from datasets import load_dataset
import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import CreateBucketDetails
from streaming import MDSWriter
from lightning import Trainer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for MosaicML Streaming."
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Path where we want to store dataset pre-conversion.")
    parser.add_argument("--dataset-name", default="tau/scrolls",
                        help="Dataset to pull from huggingface.")
    parser.add_argument("--dataset-subname", default="narrative_qa",
                        help="The sub-dataset in tau/scrolls. Not all datasets have this")
    parser.add_argument("--streaming", action="store_true", help="Stream large datasets, rather than pull all")
    parser.add_argument("--mds-output-bucket", required=True,
                        help="Remote OCI output bucket to write reformatted MDS data to.")
    parser.add_argument("--compartment-id", required=True,
                        help="The compartment ID in OCI in which to create the bucket")
    return parser.parse_args()

def create_bucket_if_not_exists(
        object_storage_client: ObjectStorageClient,
        bucket_name: str,
        namespace: str,
        oci_compartment_id: str
    ) -> None:
    """ Creates output bucket in OCI if it does not already exist.
    Args:
        object_storage_client: oci.object_storage.ObjectStorageClient to connect to object storage
        bucket_name: Name of output bucket to check or create
        namespace: Object storage namespace (retrieved prior to this in main function)
        oci_compartment_id: Compartment ID in OCI in which to create the bucket.
    Returns: None
    """
    try:
        object_storage_client.get_bucket(namespace_name=namespace, bucket_name=bucket_name)
        print(f"Bucket {bucket_name} already exists")
    except oci.exceptions.ServiceError as e:
        if e.status == 404: # Bucket DNE
            print(f"Creating bucket: {bucket_name} in namespace: {namespace}")
            create_bucket_details = CreateBucketDetails(
                name=bucket_name,
                compartment_id=oci_compartment_id,
                storage_tier="Standard",
                public_access_type="NoPublicAccess"
            )
            object_storage_client.create_bucket(namespace_name=namespace,
                                                create_bucket_details=create_bucket_details)
            print(f"Bucket {bucket_name} created successfully.")
        else:
            raise

def main():
    ## Perform format conversion for dataset.
    args = parse_args()
    data = load_dataset(
        args.dataset_name,
        name=args.dataset_subname,
        trust_remote_code=True,
        split="train",
        cache_dir=args.dataset_path,
        streaming=args.streaming
    )

    dtype_mapping = {"string": "str"}
    columns = {}
    for feature_name, feature in data.features.items():
        columns[feature_name] = dtype_mapping[feature.dtype]

    config = oci.config.from_file()
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    create_bucket_if_not_exists(object_storage_client=object_storage_client,
                                bucket_name=args.mds_output_bucket,
                                namespace=namespace,
                                oci_compartment_id=args.compartment_id)
    remote_bucket = f'oci://{args.mds_output_bucket}@{namespace}/'
    with MDSWriter(out=remote_bucket, columns=columns) as writer:
        for item in data:
            writer.write(item)

    print(f"Successfully wrote mds output to {remote_bucket}")

    

if __name__ == "__main__":
    sys.exit(main())
    