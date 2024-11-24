import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sys

from datasets import load_dataset
import oci
from oci.object_storage import ObjectStorageClient
import webdataset as wds
from tqdm import tqdm

# Local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.oci_utils import create_bucket_if_not_exists 
from utils.parse_args import hf_optimizer_parse_args

# Function to upload data to OCI Object Storage
def upload_to_oci(object_storage_client: ObjectStorageClient, namespace, bucket_name, object_name, data, fname):
    print(f"Uploading {fname} to as {object_name} to {bucket_name}")
    object_storage_client.put_object(
        namespace,
        bucket_name,
        object_name,
        data
    )
    os.remove(fname)



# Function to prepare each sample for WebDataset
def process_sample(sample, index):
    # Create an in-memory dictionary with the data
    data = {
        "__key__": f"{index:06d}",  # Use a formatted index as the key
        "text": sample["text"]       # Store the text under the "txt" key
    }
    return data

async def async_upload_to_oci(executor, object_storage_client, namespace, bucket_name, object_name, data, fname):
    # Run the upload function in a separate thread
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, upload_to_oci, object_storage_client, namespace, bucket_name, object_name, data, fname)

async def process_and_upload(dataset, shard_pattern, executor, object_storage_client, namespace, bucket_name, compress=False, max_uploaders=1):
    upload_tasks = set()
    with wds.ShardWriter(shard_pattern, verbose=True, maxcount=2**16, maxsize=2**26, compress=compress) as sink:
        for index, sample in enumerate(tqdm(dataset)):
            # Process and write each sample
            data = process_sample(sample, index)
            sink.write(data)

            # Check if the current shard is complete
            if sink.count >= sink.maxcount or sink.size >= sink.maxsize:
                # Convert the shard to a BytesIO object
                current_shard = sink.fname
                sink.next_stream()

                # Upload the shard to OCI Object Storage
                upload_task = asyncio.create_task(
                    async_upload_to_oci(
                        executor=executor,
                        object_storage_client=object_storage_client,
                        namespace=namespace,
                        bucket_name=bucket_name,
                        object_name=os.path.basename(current_shard),
                        data=open(current_shard, 'rb').read(),
                        fname=current_shard
                    )
                )
                upload_tasks.add(upload_task)
                if len(upload_tasks) > max_uploaders:
                    done, upload_tasks = await asyncio.wait(upload_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        await task
    return upload_tasks


async def main():
    args = hf_optimizer_parse_args()
    executor = ThreadPoolExecutor(max_workers=args.max_workers)  # Adjust the number of workers as needed
    # Load the C4 dataset from AllenAI
    dataset = load_dataset(args.dataset_name,
                           args.dataset_subname,
                           trust_remote_code=True,
                           split="train",
                           streaming=True)

    # Set up OCI client
    config = oci.config.from_file(file_location=args.oci_config_path,
                                  profile_name=args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    create_bucket_if_not_exists(object_storage_client=object_storage_client,
                                bucket_name=args.output_bucket, namespace=namespace,
                                oci_compartment_id=args.compartment_id)

    # Initialize the ShardWriter to write to an in-memory buffer
    cache_dir = os.path.expanduser(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    shard_pattern = os.path.join(cache_dir, "c4_shard-%06d.tar")
    all_upload_tasks = await process_and_upload(
        dataset=dataset,
        shard_pattern=shard_pattern,
        executor=executor,
        object_storage_client=object_storage_client,
        namespace=namespace,
        bucket_name=args.output_bucket,
        compress=args.compress_data,
        max_uploaders=args.max_workers
    )

    if all_upload_tasks:
        await asyncio.gather(*all_upload_tasks)
    executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())