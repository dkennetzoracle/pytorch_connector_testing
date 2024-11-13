# PyTorch Connector Testing

This repo contains subdirectories which test various PyTorch connectors for loading data from OCI object storage.

When working with PyTorch connectors pulling data from object storage, it is generally recommended by each connector to convert the dataset into an optimized format for cloud storage (generally larger shards that are more efficient to load). If provided, the tests will follow the conversion of data to the optimized format prior to running the test so that we are following best practices.

Given a test and dataset, `transformers.Trainer()` will be used with LoRA peft for fine-tuning. The `transformers.Trainer()` will use a custom callback function to save checkpoints, where it will directly write checkpoints to OCI object storage. The general testing flow will look like the following:

1. Optimize the dataset using the library specific optimizer in OCI object storage.
2. Stream the dataset from OCI object storage during training
3. Write the model checkpoints using a custom callback which uses the connector's writing functionality (if provided)

## Evaluations

Both technical and experiential evaluations will be performed on each connector. In this way, if connectors share similar performance characteristics, experiential evaluations may guide preference.

### Experiential Evaluation Criteria
The following experiential evaluations will be made with respect to each connector, as they are important for adoption:
- **Ease of use**: how much work / studying / customization do I need to get each connector to work
- **Flexibility** - is it flexible to handle a variety of data types
- **Integration** - How well does it integrate with existing / common tooling?
- **Popularity** - Is it easy to find information / example usage? Does it have a lot of GitHub stars?
- **Documentation** - How good is the documentation for the tooling

### Technical Evaluation Criteria
The following technical evaluations will be made with respect to each connector as performance measurements:

| Metric | Description |
| :---   | :---        |
| Total Time (s) | End to end time of fine-tuning|
| GPU % Utilization | Percent of GPU compute utilized  |
| CPU % Utilization | Percent of CPU compute utilized  |
| GPU % Memory Utilization | Percent of GPU memory utilized |
| CPU % memory Utilization | Percent of CPU memory utilized |