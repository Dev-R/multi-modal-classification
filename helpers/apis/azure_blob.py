import os

from azure.storage.blob import (
    BlobServiceClient,
    __version__,
)


AZURE_STORGE_ACCOUNT = "multimodalmedia "
VIDEO_CONTAINER_NAME = "video-uploads"
IMAGE_CONTAINER_NAME = "ima-uploads"

# connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
connect_str = ""


def get_media_path(img_name) -> str:
    """Find an media path in system and return it"""
    # Get media directory
    medias_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    # Get media path
    media_path = os.path.join(medias_dir, img_name)
    # Return media path
    return media_path


def upload_blob(filename: str, is_video: bool) -> str:
    """
    Upload a file to an Azure blob container and return the URL.

    Args:
        filename (str): The name of the file to be uploaded.
        is_video (bool): Indicates whether the file is a video or not.

    Returns:
        str: The URL of the uploaded file in the Azure blob container.

    Raises:
        Exception: If an error occurs during the upload.
    """
    container_name = IMAGE_CONTAINER_NAME
    if is_video:
        container_name = VIDEO_CONTAINER_NAME
    try:
        # Create the BlobServiceClient object which will be used to create a container client
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        upload_file_path = get_media_path(filename)
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=filename
        )
        # Upload the created file
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data)
        # return file blob url
        file_url = (
            "https://"
            + AZURE_STORGE_ACCOUNT
            + ".blob.core.windows.net"
            + "/"
            + container_name
            + "/"
            + filename
        )
        return file_url.replace(" ", "")

    except Exception as ex:
        print("Exception:")
        print(ex)
        print("filename" + filename)

    # print("get_media_path", get_media_path('video'))
