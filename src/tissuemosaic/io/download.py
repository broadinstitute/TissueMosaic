import os
from google.cloud import storage


def download_from_bucket(bucket_name: str, source_path: str, destination_path: str):
    """
    Helper function to download files from google buckets.

    Args:
        bucket_name: the name of the google bucket. For example: "ld-data-bucket"
        source_path: path to the file in the bucket. For example "tissue-mosaic/slideseq_testis_anndata_h5ad.tar.gz"
        destination_path: path in the local filesystem to save file. For example "my_dir/my_file_h5ad.tar.gz"
    """

    # create the directory where the file will be written
    dirname_tmp = os.path.dirname(destination_path)
    os.makedirs(dirname_tmp, exist_ok=True)

    # connect ot the google bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_path, bucket_name, destination_path
        )
    )


