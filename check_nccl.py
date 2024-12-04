import torch
import torch.distributed as dist
from accelerate import Accelerator

def main():
    # Initialize the accelerator
    accelerator = Accelerator()

    # Get the current rank and world size
    rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes

    # Print the rank and world size
    print(f"Rank {rank}/{world_size} is running.")

    # Each process creates a tensor with its rank
    tensor = torch.tensor([rank], dtype=torch.bfloat16).to(accelerator.device)

    # Perform a simple operation: all-reduce the tensor
    dist.all_reduce(tensor)

    # Print the result from each process
    print(f"After all_reduce, Rank {rank} has tensor value: {tensor.item()}")

if __name__ == "__main__":
    main()