import argparse
import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import sys
import os
from datetime import datetime

import oci

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.oci_utils import create_bucket_if_not_exists 

#-------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('raw_optimizer.log')]
)
logger = logging.getLogger('raw_optimizer')
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
def write_n_examples_to_new_file_and_upload(data_iterator,
                                            target_bucket,
                                            prefix,
                                            object_storage: oci.object_storage.ObjectStorageClient,
                                            namespace,
                                            N):
    """
    Writes every N JSON objects to a new JSONL file and uploads it to the target bucket.
    """
    buffer = []
    chunk_index = 0

    for record in data_iterator:
        buffer.append(record)
        if len(buffer) == N:
            file_name = f"{prefix}.{chunk_index:05d}.json.gz"
            chunk_data = "\n".join(json.dumps(obj) for obj in buffer).encode('utf-8')
            compressed_data = gzip_data(chunk_data)
            object_storage.put_object(
                namespace,
                target_bucket,
                file_name,
                compressed_data
            )
            buffer = []
            chunk_index += 1

    # Upload any remaining records
    if buffer:
        file_name = f"{prefix}_part_{chunk_index}.jsonl"
        chunk_data = "\n".join(json.dumps(obj) for obj in buffer).encode('utf-8')
        object_storage.put_object(
            namespace,
            target_bucket,
            file_name,
            chunk_data
        )
    logger.info(f"Uploaded {chunk_index + 1} chunks for {prefix}.*.json.gz")


#-------------------------------------------------------------------------------
def process_file(source_bucket, target_bucket, file_name, object_storage, namespace, N):
    """
    Downloads, processes, and uploads data in chunks.
    """
    data_iterator = download_and_unzip_file(source_bucket, file_name, object_storage, namespace)
    write_n_examples_to_new_file_and_upload(data_iterator,
                                            target_bucket,
                                            file_name.split(".json.gz")[0],
                                            object_storage, namespace, N)


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
    create_bucket_if_not_exists(object_storage, source_bucket, namespace, args.compartment_id)
    create_bucket_if_not_exists(object_storage, target_bucket, namespace, args.compartment_id)
    file_names = get_object_details(object_storage, namespace, source_bucket)

    N = args.num_samples_per_shard  # Number of records per file

    start_time = datetime.now()
    logger.info(f"START TIME: {start_time.strftime('%A, %d %B, %Y, %H:%M:%S')}")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, source_bucket, target_bucket, file_name, object_storage, namespace, N)
            for file_name in file_names
        ]
        for future in futures:
            future.result()  # Wait for completion

    end_time = datetime.now()
    logger.info(f"END TIME: {end_time.strftime('%A, %d %B, %Y, %H:%M:%S')}")
    logger.info(f"Total Duration: {end_time - start_time}")


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
