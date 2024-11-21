from litdata import StreamingDataset, StreamingDataLoader

import torch
import pyarrow as pa

def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, pa.ChunkedArray):
        return torch.tensor(elem.to_numpy())
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, dict):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, (tuple, list)):
        return type(elem)(custom_collate(samples) for samples in zip(*batch))
    else:
        return torch.utils.data._utils.collate.default_collate(batch)

# Create the Streaming Dataset
dataset = StreamingDataset('/mnt/nvme/datasets/litdata/', shuffle=True)

# Create a DataLoader
dataloader = StreamingDataLoader(dataset, batch_size=32, collate_fn=custom_collate)

count = 0
# Iterate over the dataset
for batch in dataloader:
    # Process your batch here
    count +=1
print(f"{count=}")