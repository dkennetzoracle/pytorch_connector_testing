# Mosaic ML Streaming

[MosaicML Streaming](https://github.com/mosaicml/streaming) contains a `StreamingDataset` to make training on large datasets from cloud storage as fast, cheap, and scalable as possible. `StreamingDataset` is compatible with any data type, including images, text, video, and multimodal data.

It also natively supports multi-cloud, specifically listing OCI as compatible.

Testing is performed on the BM.H100.8 shape in OCI, which is a single node with 8 H100 NVIDIA GPUs. This walk-through can be performed with a smaller dataset and model as an example. However, to replicate the benchmarks performed, this setup should be used.

## Tests
1. Pull data from hf and convert to optimized format while sending to bucket (automated conversion) - `01_pull_data_and_upload_to_bucket.py`
2. Pull data from hf and convert to optimized + compressed format while sending to bucket (automated conversion) - `01_pull_data_and_upload_to_bucket.py`
3. Measure size and object count uncompressed - `02_get_count_size_time_bucket.py`
4. Measure size and object count compressed - `02_get_count_size_time_bucket.py`
5. Stream dataset from object storage into fine-tuning - measure test metrics uncompressed - write to local storage
6. Stream dataset from object storage into fine-tuning - measure test metrics compressed - write to local storage

### Setup
Setup is simple, and requires an OCI account with permission to create buckets in your tenancy / compartment. To setup your account for cli / sdk access, see [here](provide_link). You will need your compartment ID in subsequent steps. To find it visit [this page](insert_link).

```bash
python3 -m venv venv
source venv bin activate
pip3 install -r requirements.txt
```

### Step 1 & 2 - Stream data from HF and upload optimized file format directly to OCI object storage
Using a large text dataset [allenai/c4 en](https://huggingface.co/datasets/allenai/c4), we will stream data from huggingface and optimize to Mosaic Data Shard (MDS) format which is the preferred format for streaming with MosaicML. We will run tests for both compressed and uncompressed data, to see the impact of compression on both storage and performance. Smaller objects are faster to transfer and cheaper to store, but compression and decompression both come with a cost - so we will explore that!

Since MDS writer gives us access to multi-threaded uploads, we will take advantage of that utilizing a moderate number of threads for uploading.

Note: `--dataset-path` can be small since data in the example will be streamed for huggingface, so it will not be stored locally.

Uncompressed upload - this will take several hours:
```bash
python3 01_pull_data_and_upload_to_bucket.py \
--dataset-path /path/to/cache \
--dataset-name allenai/c4 \
--dataset-subname en \
--stream \
--mds-output-bucket mosaic_ml_allenai_c4_en \
--compartment-id ocid1.compartment.oc1..aaa...asdf \
--max-workers 32
```

Compressed upload - this will take several hours:
```bash
python3 01_pull_data_and_upload_to_bucket.py \
--dataset-path /path/to/cache \
--dataset-name allenai/c4 \
--dataset-subname en \
--stream \
--mds-output-bucket mosaic_ml_allenai_c4_en_compressed \
--compartment-id ocid1.compartment.oc1..aaa...asdf \
--max-workers 32
```

### Step 3:
```bash
python3 03_stream_data_from_bucket.py \
--compartment-id ocid1.compartment.oc1..aaaaaaaa...123a \
--local-cache /mnt/nvme/datasets/mds_cache \
--bucket-name mosaic_ml_allenai_c4_en_compressed
```

## Experiential Evaluations

### Dataset optimization
The library is very well documented and very easy to use. It was as simple as pulling data from huggingface in streaming mode and writing with their MDS writer. This gets a 10/10 because "it just works". I am actually very impressed - I simply pull data from HF in streaming mode, and the dataset writer converts those streams to Mosaic Data Shards and writes them directly to my object storage.



