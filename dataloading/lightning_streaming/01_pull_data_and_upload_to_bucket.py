from pyarrow import json as pajson
from pathlib import Path
from litdata import optimize

def process_json(filepath_index):
    filepath, index = filepath_index
    data = pajson.read_json(filepath)
    return data["text"], index

if __name__ == "__main__":
    input_dir = Path("/mnt/nvme/datasets/allenai/c4_en/small/")

    json_files = sorted(list(input_dir.rglob("c4-train*")))
    indexed_json_files = [(path, index) for index, path in enumerate(json_files)]
    outputs = optimize(
        fn=process_json,
        inputs=indexed_json_files,
        output_dir="/mnt/nvme/datasets/litdata",
        chunk_size=64 * 1024 * 1024
    )