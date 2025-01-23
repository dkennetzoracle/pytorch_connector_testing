import torch

num_devices = torch.cuda.device_count()

for i in range(num_devices):
    name = torch.cuda.get_device_name(i)
    device_properties = torch.cuda.get_device_properties(i)
    print(f"Device {i}: {name}")
    print(f"  - Compute Capability: {device_properties.major}.{device_properties.minor}")
    print(f"  - Total Memory: {device_properties.total_memory / (1024 ** 3):.2f} GB")
    print(f"  - Multiprocessors: {device_properties.multi_processor_count}")
