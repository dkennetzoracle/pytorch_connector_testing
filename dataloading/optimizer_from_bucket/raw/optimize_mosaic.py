import argparse
import gzip
import json
import logging
from io import BytesIO
import sys
import os
from datetime import datetime

import oci

from streaming import MDSWriter

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.oci_utils import create_bucket_if_not_exists 

#-------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('mosaic_optimizer.log')]
)
logger = logging.getLogger('mosaic_optimizer')
logger.setLevel(logging.INFO)


#-------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="default optimizer for sharding json data.")
    parser.add_argument("-i", "--input-bucket", required=True,
                        help="input bucket with json data to shard.")
    parser.add_argument("-o", "--output-bucket", required=True,
                        help="Output bucket to write sharded data to.")
    parser.add_argument("-m", "--compartment-id", required=True,
                        help="compartment id to use.")
    parser.add_argument("-c", "--config", default="~/.oci/config",
                        help="oci config to use.")
    parser.add_argument("-p", "--profile", default="DEFAULT",
                        help="oci profile to use.")
    parser.add_argument("-n", "--num-samples-per-shard", default=50000,
                        help="samples per shard.")
    parser.add_argument("-w", "--max-workers", type=int, default=1,
                        help="num additional workers for data.")
    return parser.parse_args()


#-------------------------------------------------------------------------------
def download_and_unzip_file(bucket_name,
                            file_name,
                            object_storage: oci.object_storage.ObjectStorageClient,
                            namespace):
    """
    Downloads and unzips a gzipped JSONL file from an OCI Object Storage bucket.
    Reads the file line by line and yields individual JSON objects.
    """
    logger.info(f"Downloading file: {file_name} from bucket: {bucket_name}")
    response = object_storage.get_object(namespace, bucket_name, file_name)
    compressed_data = response.data.content
    with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
        for line in f:
            yield json.loads(line)


#-------------------------------------------------------------------------------
def gzip_data(data):
    """
    Compresses the given data using gzip and returns it as bytes.
    """
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
        gz.write(data)
    return buffer.getvalue()


#-------------------------------------------------------------------------------
def write_n_examples_to_mds(data_iterator,
                            writer):
    """
    Writes every N JSON objects to a new JSONL file and uploads it to the target bucket.
    """
    for record in data_iterator:
        writer.write(record)



#-------------------------------------------------------------------------------
def process_file(source_bucket, target_bucket, file_name, object_storage, namespace, writer):
    """
    Downloads, processes, and uploads data in chunks.
    """
    logger.info(f"Processing file: {file_name}")
    data_iterator = download_and_unzip_file(source_bucket, file_name, object_storage, namespace)
    write_n_examples_to_mds(data_iterator,
                            writer)
    logger.info(f"File completed: {file_name}")


#-------------------------------------------------------------------------------
def get_object_details(object_storage_client: oci.object_storage.ObjectStorageClient,
                       namespace: str,
                       bucket_name: str,
                       pattern: str = "train"
):
    resp = object_storage_client.list_objects(
        namespace_name=namespace,
        bucket_name=bucket_name,
    )
    files = []
    for object in resp.data.objects:
        files.append(object.name)
    while resp.data.next_start_with:
        resp = object_storage_client.list_objects(namespace,
                                                  bucket_name,
                                                  start=resp.data.next_start_with)
        for object in resp.data.objects:
            files.append(object.name)
    return [f for f in files if pattern in f]


#-------------------------------------------------------------------------------
def main():
    args = parse_args()
    # Initialize OCI ObjectStorageClient
    config = oci.config.from_file(args.config, args.profile)  # Adjust config path/profile if needed
    object_storage = oci.object_storage.ObjectStorageClient(config)
    max_workers = args.max_workers if args.max_workers <= 10 else 10
    namespace = object_storage.get_namespace().data
    source_bucket = args.input_bucket
    target_bucket = args.output_bucket
    target_mosaic = f'oci://{target_bucket}@{namespace}/'
    create_bucket_if_not_exists(object_storage, target_bucket, namespace, args.compartment_id)
    file_names = get_object_details(object_storage, namespace, source_bucket)

    start_time = datetime.now()
    logger.info(f"START TIME: {start_time.strftime('%A, %d %B, %Y, %H:%M:%S')}")

    columns = {'text': 'str', 'timestamp': 'str', 'url': 'str'}
    hashes=['sha1', 'xxh64']
    size_limit = 1 << 28
    writer = MDSWriter(out=target_mosaic,
                       columns=columns,
                       compression='zstd:7',
                       size_limit=size_limit,
                       hashes=hashes,
                       max_workers=max_workers)

    for file_name in file_names:
        process_file(source_bucket, target_mosaic, file_name,
                     object_storage, namespace, writer)
    writer.finish()
    end_time = datetime.now()
    logger.info(f"END TIME: {end_time.strftime('%A, %d %B, %Y, %H:%M:%S')}")
    logger.info(f"Total Duration: {end_time - start_time}")


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
