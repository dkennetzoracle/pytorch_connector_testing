# Mosaic ML Streaming

[MosaicML Streaming](https://github.com/mosaicml/streaming) contains a `StreamingDataset` to make training on large datasets from cloud storage as fast, cheap, and scalable as possible. `StreamingDataset` is compatible with any data type, including images, text, video, and multimodal data.

It also natively supports multi-cloud, specifically listing OCI as compatible.

## Tests
1. Pull data from hf and convert to optimized format while sending to bucket (automated conversion) - measure size and object count uncompressed
2. Pull data from hf and convert to optimized + compressed format while sending to bucket (automated conversion) - measure size and object count compressed 
3. Convert a local dataset to optimized and write to bucket (less automated) - measure size and object count uncompressed
4. Convert a local dataset to optimized + compressed and write to bucket (less automated) - measure size and object count compressed
5. Stream dataset from object storage into fine-tuning - measure test metrics uncompressed
6. Stream dataset from object storage into fine-tuning - measure test metrics compressed



## Experiential Evaluations

### Dataset optimization
The library is very well documented and very easy to use. It was as simple as pulling data from huggingface in streaming mode and writing with their MDS writer. This gets a 10/10 because "it just works". I am actually very impressed - I simply pull data from HF in streaming mode, and the dataset writer converts those streams to Mosaic Data Shards and writes them directly to my object storage.



