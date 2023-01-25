import replicate

# API documentation: https://replicate.com/docs/get-started/python
# Model documentation: https://replicate.com/nightmareai/real-esrgan/api

TOKEN = "4d7009bbb7c63b6cc9a977d9d614f1050429e4a4"


def get_model_client(token):
    """
    Returns a replicate Client object with the specified token.

    Parameters:
    - token (str): the API token to use for authentication.

    Returns:
    - A replicate.Client object.
    """
    return replicate.Client(token)


def get_model(client, model_name):
    """
    Returns a model object with the specified name.

    Parameters:
    - client (replicate.Client): the Client object to use for interacting with the API.
    - model_name (str): the name of the model to retrieve.

    Returns:
    - A replicate.Model object.
    """
    models = client.models
    return models.get(model_name)


def get_model_version(model, version_id):
    """
    Returns a model version object with the specified ID.

    Parameters:
    - model (replicate.Model): the model object to use for retrieving the version.
    - version_id (str): the ID of the model version to retrieve.

    Returns:
    - A replicate.ModelVersion object.
    """
    versions = model.versions
    return versions.get(version_id)


def run_prediction(version, image_path):
    """
    Runs a prediction on the specified model version with the specified image.

    Parameters:
    - version (replicate.ModelVersion): the model version object to use for running the prediction.
    - image_path (str): the path to the image to use for the prediction.

    Returns:
    - The prediction result.
    """
    inputs = {
        # Input image
        'image': open(image_path, "rb"),
        # Factor to scale image by
        # Range: 0 to 10
        "scale": 10,
        # Face enhance
        "face_enhance": False,
    }
    return version.predict(**inputs)


def apply_esragan(img_path: str) -> dict:
    """
    Runs ESRAGAN model on the specified image

    Args:
        img_path (str): the path to the image to use for the prediction.

    Return:
    - Dict contains enhance image URL
    """
    # Create the client
    client = get_model_client(TOKEN)

    # Get the model
    model = get_model(client, "nightmareai/real-esrgan")

    # Get the model version
    version = get_model_version(
        model, "42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b"
    )

    # Run the prediction
    # path_to_image = 'inputs/steve.png'
    url = run_prediction(version, img_path)

    # Print the result
    print({"url": url})

    # Return result if any
    return url

if __name__ == "__main__":
    apply_esragan()
