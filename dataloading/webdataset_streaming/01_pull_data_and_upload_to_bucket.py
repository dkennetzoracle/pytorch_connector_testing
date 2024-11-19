import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import CreateBucketDetails
import os
import webdataset as wds
from datasets import load_dataset
from tqdm import tqdm
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

async def process_and_upload(dataset, shard_pattern, executor, object_storage_client, namespace, bucket_name):
    upload_tasks = set()
    with wds.ShardWriter(shard_pattern, verbose=True, maxcount=10000, maxsize=2**26) as sink:
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
                if len(upload_tasks) > 10:
                    done, upload_tasks = await asyncio.wait(upload_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        await task
    return upload_tasks


async def main():
    executor = ThreadPoolExecutor(max_workers=4)  # Adjust the number of workers as needed
    # Load the C4 dataset from AllenAI
    dataset = load_dataset("allenai/c4", "en", split="train", trust_remote_code=True, cache_dir="/mnt/nvme/datasets/allenai_c4_en", streaming=True)
    compartment_id = "ocid1.compartment.oc1..aaaaaaaa5rwhi5wj3grdiqzvz244gwzycpfl2ctlb4nvl7vi7wu55tqi375a"
    # Set up OCI client
    config = oci.config.from_file("~/.oci/config")
    object_storage_client = oci.object_storage.ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    bucket_name = "wds_allenai_c4_en_compressed"
    create_bucket_if_not_exists(object_storage_client=object_storage_client,
                                bucket_name=bucket_name, namespace=namespace,
                                oci_compartment_id=compartment_id)

    # Initialize the ShardWriter to write to an in-memory buffer
    cache_dir = "/mnt/nvme/datasets/wds_shards/"
    os.makedirs(cache_dir, exist_ok=True)
    shard_pattern = os.path.join(cache_dir, "c4_shard-%06d.tar")
    all_upload_tasks = await process_and_upload(
        dataset=dataset,
        shard_pattern=shard_pattern,
        executor=executor,
        object_storage_client=object_storage_client,
        namespace=namespace,
        bucket_name=bucket_name
    )

    if all_upload_tasks:
        await asyncio.gather(*all_upload_tasks)
    executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())