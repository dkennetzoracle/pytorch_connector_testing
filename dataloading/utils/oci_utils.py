import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import CreateBucketDetails

def create_bucket_if_not_exists(
        object_storage_client: ObjectStorageClient,
        bucket_name: str,
        namespace: str,
        oci_compartment_id: str
    ) -> None:
    """ Creates output bucket in OCI if it does not already exist.
    Args:
        object_storage_client: oci.object_storage.ObjectStorageClient to connect to object storage
        bucket_name: Name of output bucket to check or create
        namespace: Object storage namespace (retrieved prior to this in main function)
        oci_compartment_id: Compartment ID in OCI in which to create the bucket.
    Returns: None
    """
    try:
        object_storage_client.get_bucket(namespace_name=namespace, bucket_name=bucket_name)
        print(f"Bucket {bucket_name} already exists")
    except oci.exceptions.ServiceError as e:
        if e.status == 404: # Bucket DNE
            print(f"Creating bucket: {bucket_name} in namespace: {namespace}")
            create_bucket_details = CreateBucketDetails(
                name=bucket_name,
                compartment_id=oci_compartment_id,
                storage_tier="Standard",
                public_access_type="NoPublicAccess"
            )
            object_storage_client.create_bucket(namespace_name=namespace,
                                                create_bucket_details=create_bucket_details)
            print(f"Bucket {bucket_name} created successfully.")
        else:
            raise