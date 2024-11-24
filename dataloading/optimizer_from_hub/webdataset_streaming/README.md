
1. Optimize:
```bash
python3 ./01_pull_data_and_upload_to_bucket.py \
--dataset-name allenai/c4 \
--dataset-subname en \
--cache-dir /mnt/nvme/datasets/wds_shards/ \
--compress-data \
--output-bucket wds_hub_allenaic4en 
--compartment-id ocid1.compartment.oc1..aaaaaaaa...123a \
--max-workers 32
```
