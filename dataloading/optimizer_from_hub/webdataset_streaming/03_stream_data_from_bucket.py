import os
from shutil import rmtree
import sys
from time import perf_counter
from typing import List

import webdataset as wds
from tqdm import tqdm

import oci
from oci.object_storage import ObjectStorageClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from utils.parse_args import stream_from_bucket_parser
from utils.silly_utils import count_lists_in_key

def list_all_objects_in_bucket(object_storage_client: ObjectStorageClient, namespace: str, bucket: str, par: str, cache: str) -> List[str]:
    objects = []
    resp = object_storage_client.list_objects(namespace_name=namespace, bucket_name=bucket)
    for object in resp.data.objects:
        objects.append(f'{par}{object.name}')
    
    while resp.data.next_start_with:
        resp = resp = object_storage_client.list_objects(namespace, bucket, start=resp.data.next_start_with)
        for object in resp.data.objects:
            objects.append(f'{par}{object.name}')

    return [f"pipe: curl -L -s {object}" for object in objects if object.endswith(".tar")]
    #return [f"pipe: curl -L {object}" for object in objects]
    #return [f"pipe:aws s3 cp s3://{os.path.join('optimized-imagenet-1m', key)}  -" for key in keys if "train" in key]

def main():
    args = stream_from_bucket_parser()
    if not args.pre_authenticated_request:
        raise ValueError("Pre-authenticated request required for WDS.")
    if os.path.isdir(args.local_cache):
        rmtree(args.local_cache)
    os.makedirs(args.local_cache, exist_ok=True)
    config = oci.config.from_file(file_location=args.oci_config_path,
                                  profile_name=args.oci_profile)
    object_storage_client = ObjectStorageClient(config)
    namespace = object_storage_client.get_namespace().data
    objects = list_all_objects_in_bucket(object_storage_client=object_storage_client,
                                         namespace=namespace,
                                         bucket=args.bucket_name,
                                         par=args.pre_authenticated_request,
                                         cache=args.local_cache)
    dataset = wds.WebDataset(objects, shardshuffle=True).shuffle(1000)

    data_loader = wds.WebLoader(dataset, batch_size=64, num_workers=64)
    start = perf_counter()
    total_samples = 0
    log_file = args.log_file if args.log_file else "wds_log.txt"
    out = ""
    for epoch in range(2):
        num_samples = 0
        t0 = perf_counter()

        for data in tqdm(data_loader, smoothing=0, mininterval=1):
            num_samples += count_lists_in_key(data, 'text')
            if num_samples >= 1000000:
                break

        print(f"Completed epoch {epoch}")
        total_samples += num_samples
        t1 = perf_counter()
        out += f"For {__file__} on {epoch}, streamed over {num_samples} samples in {t1 - t0}s or {num_samples / (t1 - t0)} samples / sec.\n"
    end = perf_counter()
    out += f"For {__file__}: streamed {total_samples} in {end - start}s or {total_samples / (end - start)} samples / sec."
    with open(log_file, 'w') as f:
        f.write(out)

if __name__ == "__main__":
    sys.exit(main())