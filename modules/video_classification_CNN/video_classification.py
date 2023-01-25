"""
    Optimized version of action.py with no training models / data
    TODO:
        Add argv to model so user can choose to live model or on video
        I.e: 
            Case 1: python video_classification.py (Default, will run live video: if cam not found raise error) 
            Case 2: python video_classification.py --url wwww.youtube.com/... (Not default, user has to pass a valid youtube url)
    TODO: 
        Make two functions one for live predection and one for non-live
"""
import os
import cv2
import numpy as np
from moviepy.editor import *
from collections import deque
from tensorflow import keras
import youtube_dl
import argparse

"""
    - Introduction to Video Classification and Human Activity Recognition:
        * https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/

"""


from tensorflow.keras.layers import *


IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_IMAGES_PER_CLASS = 8000

DATASET_DIRECTORY = "modules/video_classification_CNN/UCF_CRIME"
CLASSES_LIST = os.listdir(DATASET_DIRECTORY)  # Return all classes name in UCF50


def download_youtube_videos(url, output_dir):
    # Create a dictionary of options to pass to youtube_dl
    ydl_opts = {
        # Specify the output template for the downloaded file.
        # %(title)s will be replaced with the title of the video.
        # %(ext)s will be replaced with the file extension of the video.
        "outtmpl": output_dir + "/%(title)s.%(ext)s",
        # Set format to download the video in MP4 format
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        # Set quiet to True to suppress output messages.
        "quiet": True,
    }
    # Create a YoutubeDL object with the options specified above
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        # Extract information about the video and download it
        info = ydl.extract_info(url, download=True)
        # Get the title of the video
        video_title = info.get("title", None)
    # Return the title of the downloaded video
    return video_title


def predict_on_live_video(
    model: str,
    video_file_path: str = None,
    output_file_path: str = None,
    window_size: int = 25,
) -> None:
    """
        This function uses a trained model to predict on live video from the default webcam.
    Args:
        model (str): trained model file path.
        video_file_path (str, optional): path to the video file. Defaults to None.
        output_file_path (str, optional): path to the output file. Defaults to None.
        window_size (int, optional): size of the window for moving/rolling average. Defaults to 25.
    Returns:
        None
    """
    cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Video", 800, 600)

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)

    model = keras.models.load_model(model)

    if video_file_path:
        print(
            "Calling the predict_on_live_video method on non-live to start the Prediction."
        )
        # Reading the Video File using the VideoCapture Object
        video_reader = cv2.VideoCapture(video_file_path)
    else:
        print(
            "Calling the predict_on_live_video method on live to start the Prediction."
        )
        # Initialize the VideoCapture object to read from the default webcam
        video_reader = cv2.VideoCapture(0)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(
        output_file_path,
        cv2.VideoWriter_fourcc("M", "P", "4", "V"),
        24,
        (original_video_width, original_video_height),
    )
    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(
            np.expand_dims(normalized_frame, axis=0)
        )[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(
                predicted_labels_probabilities_deque
            )

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = (
                predicted_labels_probabilities_np.mean(axis=0)
            )

            # Get the top 5 predicted classes
            top_5_indices = np.argsort(predicted_labels_probabilities_averaged)[-5:]
            top_5_indices = top_5_indices[::-1]
            # Get the class names for the top 5 predicted classes
            top_5_class_names = [CLASSES_LIST[i] for i in top_5_indices]

            # Print the top 5 class names
            print("Top 5 predicted classes: ", top_5_class_names)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = CLASSES_LIST[predicted_label]

            # TODO: Add  FPS to puttext
            # Overlaying Class Name Text Ontop of the Frame

            # Loop through the top 5 class names and display them on the frame
            for i, class_name in enumerate(top_5_class_names):
                class_confidence = round(
                    predicted_labels_probabilities_averaged[top_5_indices[i]], 3
                )
                class_name = f"{class_name}  (Confidence= {str(class_confidence)})"
                # Define the positions, font, color, and size for each class name
                positions = [
                    (10 * (i + 1), 30),
                    (160 * (i + 1), 30),
                    (10 * (i + 1), 400),
                    (90 * (i + 1), 400),
                    (2000 * (i + 1), 400),
                ]
                font = cv2.FONT_HERSHEY_SIMPLEX
                colors = [
                    (0, 255, 0),
                    (0, 165, 255),
                    (0, 255, 255),
                    (0, 0, 255),
                    (0, 0, 255),
                ]
                # Font sizes
                top_5_class_names_len = len(top_5_class_names)
                font_size = 0.55
                sizes = [font_size for _ in range(top_5_class_names_len)]
                cv2.putText(
                    frame,
                    class_name,
                    positions[i],
                    font,
                    sizes[i],
                    colors[i],
                    2,
                    cv2.LINE_AA,
                )

            # cv2.putText(
            #     frame,
            #     predicted_class_name,
            #     (10, 30),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 0, 255),
            #     2,
            # )

        # Writing The Frame
        video_writer.write(frame)
        cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Video", 1280, 1280)

        cv2.imshow("Live Video", frame)

        key_pressed = cv2.waitKey(10)

        if key_pressed == ord("q"):
            break

    cv2.destroyAllWindows()

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()
    video_writer.release()


def call_on_video(video_url):
    # Creating The Output directories if it does not exist
    print("Creating The Output directories if it does not exist  ...")
    output_directory = "Youtube_Videos"
    os.makedirs(output_directory, exist_ok=True)

    print("Downloading a YouTube Video  ...")

    # Downloading a YouTube Video
    video_title = download_youtube_videos(video_url, output_directory)
    video_title = "ABC"
    print("Getting the YouTube Video's path you just downloaded  ...")
    # Getting the YouTube Video's path you just downloaded
    input_video_file_path = f"{output_directory}/{video_title}.mp4"

    # Setting sthe Window Size which will be used by the Rolling Average Proces
    window_size = 25
    print("Constructing The Output YouTube Video Path  ...", video_title)
    # Constructing The Output YouTube Video Path
    output_video_file_path = (
        f"{output_directory}/{video_title} -Output-WSize {window_size}.mp4"
    )
    return output_video_file_path, input_video_file_path, window_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", help="URL link of Youtube video to run non-live prediction"
    )
    args = parser.parse_args()

    print("Loading model  ...")
    model = "Model___Date_Time_2023_01_15__12_21_55___Loss_0.3903857469558716___Accuracy_0.9278905987739563.h5"
    output_video_file_path, input_video_file_path = None, None
    # If args go non-live
    if args.url:
        output_video_file_path, input_video_file_path, window_size = call_on_video(
            args.url
        )

    # Calling the predict_on_live_video method to start the Prediction.
    predict_on_live_video(model, input_video_file_path, output_video_file_path)


if __name__ == "__main__":
    main()
