import argparse

def hf_optimizer_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimizer args for streaming from hf."
    )
    parser.add_argument("--oci-config-path", default="~/.oci/config",
                        help="OCI config to use. Default='~/.oci/config'")
    parser.add_argument("--oci-profile", default="DEFAULT",
                        help="OCI Profile to use. Default='DEFAULT'")
    parser.add_argument("--dataset-name", default="allenai/c4",
                        help="Dataset to pull from huggingface.")
    parser.add_argument("--dataset-subname", default="en",
                        help="The sub-dataset aka 'config'. Not all datasets have this")
    parser.add_argument("--cache-dir", required=True,
                        help="Cache for hf / webdataset. Can require a few gigs.")
    parser.add_argument("--compress-data", action="store_true",
                        help="Compress optimized data in storage")
    parser.add_argument("--output-bucket", required=True,
                        help="Remote OCI output bucket to write reformatted MDS data to.")
    parser.add_argument("--compartment-id", required=True,
                        help="The compartment ID in OCI in which to create the bucket")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max workers to use for file uploading to object storage.")
    return parser.parse_args()

def local_optimizer_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Args for local optimizer."
    )
    parser.add_argument("--oci-config-path", default="~/.oci/config",
                        help="OCI config to use. Default='~/.oci/config'")
    parser.add_argument("--oci-profile", default="DEFAULT",
                        help="OCI Profile to use. Default='DEFAULT'")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to local dataset.")
    parser.add_argument("--compress-data", action="store_true",
                        help="Compress optimized data in storage")
    parser.add_argument("--output-bucket", required=True,
                        help="Remote OCI output bucket to write reformatted MDS data to.")
    parser.add_argument("--compartment-id", required=True,
                        help="The compartment ID in OCI in which to create the bucket")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Max workers to use for file uploading to object storage.")
    return parser.parse_args()