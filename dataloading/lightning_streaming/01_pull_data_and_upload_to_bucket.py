import argparse
from itertools import islice
from functools import partial
import os
import sys

import boto3
import boto3.session
from botocore.client import Config
from datasets import load_dataset, IterableDataset
from lightning import seed_everything
import litdata as ld
from litdata.streaming import StreamingDataLoader, StreamingDataset
import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import CreateBucketDetails
from torch.utils.data.dataloader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for MosaicML Streaming."
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Path where we want to store dataset pre-conversion.")
    parser.add_argument("--compress-data", action="store_true",
                        help="Compress optimized data in storage")
    parser.add_argument("--output-bucket", required=True,
                        help="Remote OCI output bucket to write reformatted MDS data to.")
    parser.add_argument("--compartment-id", required=True,
                        help="The compartment ID in OCI in which to create the bucket")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max workers to use for file uploading to object storage.")
    parser.add_argument("--access-key", type=str, required=True,
                        help="Customer secret key name")
    parser.add_argument("--secret-key", type=str, required=True,
                        help="Customer secret key value")
    parser.add_argument("--region", type=str, required=True,
                        help="OCI region where bucket lives")
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


## Perform format conversion for dataset.
seed_everything(42)
args = parse_args()
data = load_dataset(
    args.dataset_path,
    split='train'
)

# Create bucket using OCI client, and get namespace.
config = oci.config.from_file()
object_storage_client = ObjectStorageClient(config)
namespace = object_storage_client.get_namespace().data
create_bucket_if_not_exists(object_storage_client=object_storage_client,
                            bucket_name=args.output_bucket,
                            namespace=namespace,
                            oci_compartment_id=args.compartment_id)
compression = 'zstd' if args.compress_data else None

endpoint_url = f"https://{namespace}.compat.objectstorage.{args.region}.oraclecloud.com"
bucket_endpoint = f"{endpoint_url}/{args.output_bucket}"

s3_client = boto3.client(
    's3',
    aws_access_key_id=args.access_key,
    aws_secret_access_key=args.secret_key,
    endpoint_url=endpoint_url,
    config=Config(signature_version='s3v4')
)

def optimize_fn(idx):
    return data[idx]['text']

outputs = ld.optimize(
    fn=optimize_fn,
    inputs=list(range(len(data))),
    output_dir='/mnt/nvme/datasets/litdata',
    chunk_bytes='64MB',
)

    

