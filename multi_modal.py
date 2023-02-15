# importing libraries for audio classification
import time
import threading
import cv2
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import logging
import logging
import argparse

logging.basicConfig(level=logging.INFO)

# importing libraries for video classification
from moviepy.editor import *
from multiprocessing import Process
from tensorflow.keras.layers import *

# import custom functions and params
from modules.video_classification_CNN.video_classification import predict_on_live_video
from modules.audio_classification_YAMNET.yamnet import params as params
from modules.audio_classification_YAMNET.yamnet import yamnet as yamnet_model
from modules.super_resolution_ESRGAN.real_esrgan_api import apply_esragan
from helpers.apis.azure_blob import upload_blob
from helpers.apis.twilio_api import notify_owner
from tqdm import tqdm
import helpers.helpers as helpers
from helpers.constants import (
    ActionRecognition,
    AudioRecognition,
    ObjectDetection,
    Config,
)


NOTIFICATION_ON_GOING = False
from typing import Dict, Union

# Define a type alias to represent the values in the anomaly info dictionary
AnomalyInfo = Dict[str, Union[bool, str]]

# The dictionary constant to store the anomaly information
ANOMALY_DICT = {
    "video_anomaly": {
        "is_anomaly": False,  # Whether a video anomaly has been detected
        "name": None,  # The name of the video anomaly
        "video_clip_url": None,  # The URL of the video clip where the anomaly was detected
    },
    "audio_anomaly": {
        "is_anomaly": False,  # Whether an audio anomaly has been detected
        "name": "N/A",  # The name of the audio anomaly
        "audio_clip_url": None,  # The URL of the audio clip where the anomaly was detected
    },
    "SISR": {
        "frame_exists": False,  # Whether a super-resolution frame exists
        "enhanced_frame_url": None,  # The URL of the enhanced frame
    },
    "SMS": {
        "phone_number": None,  # The phone number to send an SMS to
        "message": None,  # The message to send via SMS
    },
}

# Function to update the anomaly information for a given anomaly type
def update_anomaly_info(anomaly_type: str, info: AnomalyInfo) -> None:
    ANOMALY_DICT[anomaly_type].update(info)


# Function to retrieve the anomaly information for a given anomaly type
def get_anomaly_info(anomaly_type: str) -> AnomalyInfo:
    return ANOMALY_DICT[anomaly_type]


def get_media_path(media_name) -> str:
    """Find an media path in system and return it"""
    # Get media directory
    medias_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    # Get media path
    media_path = os.path.join(medias_dir, media_name)
    # Return media path
    return media_path


def upload_image(frame):
    """Upload image to Azure BLOB"""

    img_path_name = f"{Config.MEDIA_PATH}/frame.png"
    cv2.imwrite(img_path_name, frame)  # convert frame to image

    try:
        logging.info("Frame provided, Calling ESRGAN API ...")
        img_blob_url = apply_esragan(img_path_name)

        update_anomaly_info(
            "SISR", {"frame_exists": True, "enhanced_frame_url": img_blob_url}
        )
        logging.info("Image file uploaded ✔")
        return True
    except Exception as ex:
        logging.warning("Unable to upload image file to Azure BLOB:", ex)
        return False


def upload_video(video_name, anomaly_name):
    """Upload video to Azure BLOB"""
    try:
        logging.info("Uploading video file to Azure BLOB ...")
        media_path = get_media_path(video_name)
        video_blob_url = upload_blob(video_name, media_path, is_video=True)
        update_anomaly_info(
            "video_anomaly",
            {
                "is_anomaly": True,
                "name": anomaly_name,
                "video_clip_url": video_blob_url,
            },
        )
        logging.info("Video file uploaded ✔:" + video_blob_url)
        return True
    except Exception as ex:
        logging.info("Unable to upload video file to Azure BLOB:")
        print(ex)
        return False


def upload_media(frame, video, video_name, anomaly_name):
    is_image_uploaded = False  # Will be true if img is uploaded to cloud
    is_video_uploaded = False  # Will be true if video is uploaded to cloud

    if frame.any():
        is_image_uploaded = upload_image(frame)

    if video:
        is_video_uploaded = upload_video(video_name, anomaly_name)

    return is_image_uploaded, is_video_uploaded


def send_notification():
    """Send notification to owner"""
    logging.info("Sending notification to owner 🔔 ...")
    notify_owner(
        notification_data={
            "type": Config.NOTIFICATION_CONFIG["notification_type"],
            "data": {
                "location": "KL",
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "audio_classification": get_anomaly_info("audio_anomaly")["name"],
                "video_classification": get_anomaly_info("video_anomaly")["name"],
                "image_link": get_anomaly_info("SISR")["enhanced_frame_url"],
                "video_link": get_anomaly_info("video_anomaly")["video_clip_url"],
            },
        }
    )


def trigger_red_flag_process(anomaly_name, frame=False, video=False, video_name=None):
    """
    This function is called when an anomaly is detected in the audio or video stream. It performs the following tasks:
        1. Check if ESRGAN is already in progress, if not, set it to in progress.
        2. If a frame is provided, it converts the frame to an image, saves it to the local file system, and uploads it to Azure BLOB storage.
        3. If a video is provided, it uploads the video to Azure BLOB storage.
        4. If both image and video have been uploaded, it sends a notification to the owner with links to the image and video.

    Args:
        :param anomaly_name: The name of the anomaly detected.
        :param frame: A frame from a video stream.
        :param video: A boolean indicating if a video is provided.
        :param video_name: The name of the video file.

    Returns:
            None
    """
    global NOTIFICATION_ON_GOING
    if NOTIFICATION_ON_GOING:
        return
    NOTIFICATION_ON_GOING = True

    logging.info("Possible red flag 🚩:" + anomaly_name)
    is_image_uploaded, is_video_uploaded = upload_media(
        frame, video, video_name, anomaly_name
    )

    if is_video_uploaded and is_image_uploaded:
        send_notification()


def notification_period_daemon():
    """
    If NOTIFICATION_ON_GOING is set to true it will update
    globel NOTIFICATION_ON_GOING variable to false after t seconds

    Args:
        None

    Returns:
        None
    """

    global NOTIFICATION_ON_GOING
    logging.info("Notification daemon on going...")
    tqdm(time.sleep(Config.NOTIFICATION_CONFIG["notification_interval"]))
    NOTIFICATION_ON_GOING = False
    logging.info("Notification daemon stopped" + Config.NOTIFICATION_CONFIG)


def predict_on_live_audio():
    """
    This function predicts the live audio input using the YAMNet model.
    The model is loaded and the class names are loaded from the csv file.
    The length of the audio frames is set to 3 seconds.
    It initializes a PyAudio stream to read audio input from the microphone in real-time.
    It runs an infinite loop to process the audio and make predictions using the model.
    It plots the mel spectrogram of the audio and the model output scores for the top-score

    Args:
        None

    Returns:
        None
    """
    logging.info("Audio classification in progress......")
    # Load YAMNet model and class names
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("modules/audio_classification_YAMNET/yamnet/yamnet.h5")
    class_names = yamnet_model.class_names(
        "modules/audio_classification_YAMNET/yamnet/yamnet_class_map.csv"
    )
    # Set the length of the audio frames to 3 seconds
    frame_len = int(
        round(params.SAMPLE_RATE * params.SECONDS_OF_AUDIO_AT_A_TIME)
    )  # 2.5 sec
    # frame_len = int(params.SAMPLE_RATE * 3)  # 2.5 sec
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=params.SAMPLE_RATE,
        input=True,
        frames_per_buffer=frame_len,
    )

    # Initialize variables and start infinite loop to process audio in real-time
    cnt = 0
    TOP_N = 5
    plt.ion()
    plt.figure(figsize=(10, 6))
    while True:
        # Read audio data from the microphone
        data = stream.read(frame_len, exception_on_overflow=False)

        # Process audio data and get predictions from the model
        scores, melspec = helpers._process_and_predict(yamnet, data)
        scores_np = np.array(scores)
        mean_scores = np.mean(scores, axis=0)
        # print('mean_scores = np.mean(scores, axis=0)', mean_scores)
        top_class_indices = np.argsort(mean_scores)[::-1][:TOP_N]
        # Add anomaly name to global dict
        update_anomaly_info(
            "audio_anomaly", {"audio_clip_url": "", "name": top_class_indices[0]}
        )
        # Get the top predictions
        # Visualize the mel spectrogram of the audio
        plt.subplot(3, 1, 1)
        plt.imshow(melspec.T, aspect="auto", interpolation="nearest", origin="lower")

        # Plot and label the model output scores for the top-scoring classes.
        plt.subplot(3, 1, 3)

        plt.imshow(
            scores_np[:, top_class_indices].T,
            aspect="auto",
            interpolation="nearest",
            cmap="gray_r",
            origin="lower",
        )
        # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding - 0.5, scores.shape[0] + patch_padding - 0.5])
        # Label the TOP_N classes.
        yticks = range(0, TOP_N, 1)
        plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
        plt.pause(0.001)
        plt.show()


def predict_on_live_video(
    video_file_path: str = None,
    output_file_path: str = None,
    window_size: int = 25,
    N_MODE: bool = False,
    C_MODE: bool = False,
    STREAMING_MODE: str = "",
) -> None:
    """
    This function uses a trained model to predict on live video from the default webcam or a provided video file.
    It creates a window to display the live video and implements moving/rolling average functionality to smooth out the predictions.
    The function also saves the video with the predictions overlaid to the specified output file path.

    Args:
        model (str): trained model file path.
        video_file_path (str, optional): path to the video file. Defaults to None.
        output_file_path (str, optional): path to the output file. Defaults to None.
        window_size (int, optional): size of the window for moving/rolling average. Defaults to 25.

    Returns:
        None
    """

    # Initialize Action recogntion
    helpers._create_live_video_window()
    predicted_labels_probabilities_deque = helpers._create_rolling_average_window(
        window_size
    )

    model = helpers._load_model(ActionRecognition.MODEL)

    if video_file_path:
        video_reader = helpers._open_video_file(video_file_path)
    else:
        video_reader = helpers._open_live_webcam(STREAMING_MODE)

    video_writer, video_file_name = helpers._create_video_writer(
        video_reader, Config.MEDIA_PATH
    )

    logging.info("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
        ObjectDetection.PROTOTXT,
        ObjectDetection.MODEL,
    )

    logging.info("[INFO] Video classification in progress......")
    time.sleep(2.0)

    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        #####################################################################
        # Action Recognition
        #####################################################################
        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(
            frame, (ActionRecognition.IMAGE_HEIGHT, ActionRecognition.IMAGE_WIDTH)
        )

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255
        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(
            np.expand_dims(normalized_frame, axis=0), verbose=0
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
            top_5_class_names = [
                ActionRecognition.CLASSES_LIST[i] for i in top_5_indices
            ]

            # Define the font, color, and size for each class name
            font = cv2.FONT_HERSHEY_SIMPLEX
            colors = [
                ActionRecognition.TOP_CONFIDENCE_CLASS_TEXT_COLOR,
                ActionRecognition.HIGH_CONFIDENCE_CLASS_TEXT_COLOR,
                ActionRecognition.MID_CONFIDENCE_CLASS_TEXT_COLOR,
                ActionRecognition.LOW_CONFIDENCE_CLASS_TEXT_COLOR,
                (0, 0, 0),  # Unused slot
            ]
            font_size = (
                1.5 if STREAMING_MODE == "IP" else 0.55
            )  # If Ip cam increase font size

            sizes = [font_size for _ in range(len(top_5_class_names))]
            frame_height, frame_width, _ = frame.shape

            for i, class_name in enumerate(top_5_class_names):
                class_confidence = round(
                    predicted_labels_probabilities_averaged[top_5_indices[i]], 3
                )
                if N_MODE and i == 0:
                    class_name = f"Normal  (Confidence= {str(class_confidence)})"
                elif (
                    i == 0
                    and class_confidence > 0.8
                    and class_name != "Normal"
                    and C_MODE
                ):
                    class_name = f"Normal  (Confidence= {str(class_confidence)})"
                else:
                    class_name = f"{class_name}  (Confidence= {str(class_confidence)})"
                # else:
                #     print('Nope', i == 0 and class_confidence > 9.0, class_confidence
                #     ,
                #     class_name,
                #     class_name != "Normal"
                #     )

                # class_name = f"{class_name}  (Confidence= {str(class_confidence)})"
                current_top_class_name = top_5_class_names[0]
                text_width, text_height = cv2.getTextSize(
                    class_name, font, sizes[i], 2
                )[0]
                if i == 0:
                    text_x = 10
                    text_y = 40 + text_height
                elif i == 1:
                    text_x = frame_width - text_width - 10
                    text_y = 10 + text_height
                elif i == 2:
                    text_x = 10
                    text_y = frame_height - 50
                elif i == 3:
                    text_x = frame_width - text_width - 10
                    text_y = frame_height - 50
                elif i == 4:
                    text_x = frame_width - text_width - 10
                    text_y = (
                        frame_height + 555
                    )  # # Too LAZY, BUT WILL SET IT OFF THE SCREEN

                cv2.putText(
                    frame,
                    class_name,
                    (text_x, text_y),
                    font,
                    sizes[i],
                    colors[i],
                    2,
                    cv2.LINE_AA,
                )

                if (
                    current_top_class_name in ActionRecognition.ANOMALY_CLASSES_NAME
                    and not NOTIFICATION_ON_GOING
                    and class_confidence > ActionRecognition.ANOMALY_CLASS_CONFIDENCE
                ):
                    video_writer.release()
                    # Asynchronous method call in Python: enabled threading
                    # https://stackoverflow.com/a/1239108/16711156
                    notify = threading.Thread(
                        target=trigger_red_flag_process,
                        args=(current_top_class_name,),
                        kwargs={
                            "frame": frame,
                            "video": True,
                            "video_name": video_file_name,
                        },
                    )
                    # This will control in the background notification interval. I.e, owner will receive message every t seconds
                    timer = threading.Thread(target=notification_period_daemon)
                    timer.daemon = True
                    timer.start()
                    notify.start()

        #####################################################################
        # Object detection
        #####################################################################
        (h, w) = frame.shape[:2]
        resized_image = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(
            resized_image, (1 / 127.5), (300, 300), 127.5, swapRB=True
        )

        net.setInput(blob)
        # Predictions:
        predictions = net.forward()

        # loop over the predictions
        for i in np.arange(0, predictions.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            # predictions.shape[2] = 100 here
            confidence = predictions[0, 0, i, 2]
            # Filter out predictions lesser than the minimum confidence level
            # Here, we set the default confidence as 0.2. Anything lesser than 0.2 will be filtered
            if confidence > ObjectDetection.CONFIDENCE:
                # extract the index of the class label from the 'predictions'
                # idx is the index of the class label
                # E.g. for person, idx = 15, for chair, idx = 9, etc.
                idx = int(predictions[0, 0, i, 1])
                # then compute the (x, y)-coordinates of the bounding box for the object
                box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                # Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
                # Convert them to integers: 130 76 393 224
                (startX, startY, endX, endY) = box.astype("int")

                # Get the label with the confidence score
                label = "{}: {:.2f}%".format(
                    ObjectDetection.CLASSES[idx], confidence * 100
                )
                # logging.info("Object detected: " + label)
                # Draw a rectangle across the boundary of the object
                cv2.rectangle(
                    frame,
                    (startX, startY),
                    (endX, endY),
                    ObjectDetection.COLORS[idx],
                    2,
                )
                y = startY - 15 if startY - 15 > 15 else startY + 15
                # Put a text outside the rectangular detection
                cv2.putText(
                    frame,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    ObjectDetection.COLORS[idx],
                    2,
                )
        # Writing The Frame
        video_writer.write(frame)
        cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Video", 940, 1080)

        cv2.imshow("Live Video", frame)

        key_pressed = cv2.waitKey(10)

        if key_pressed == ord("q"):
            break

    cv2.destroyAllWindows()

    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them.
    video_reader.release()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--normal", help="the n flag", action="store_true")
    parser.add_argument(
        "-i", "--ip", help="stream video with IP camera", action="store_true"
    )
    parser.add_argument("-c", "--accuracy", help="the c flag", action="store_true")
    parser.add_argument(
        "-v", "--video", help="stream a local video", action="store_true"
    )
    video_kwargs = {
        "output_file_path": None,
        "window_size": 25,
        "C_MODE": False,
        "N_MODE": False,
        "STREAMING_MODE": "Local",  # 0, indicates web cam slot
    }
    args = parser.parse_args()

    if args.video:
        logging.info("Local Video mode activated")
        video_kwargs.update(
            {
                "video_file_path": "video.mp4",
            }
        )
    elif args.ip:
        logging.info("IP Camera mode activated")
        video_kwargs.update({"STREAMING_MODE": "IP"})
    else:
        logging.info("Web Camera mode activated as default")

    if args.normal:
        video_kwargs.update({"N_MODE": True})
        logging.info("Normal mode activate")

    if args.accuracy:
        video_kwargs.update({"C_MODE": True})
        logging.info("Accuracy mode activate")

    p1 = Process(target=predict_on_live_video, kwargs=video_kwargs)
    p1.start()
    # p2 = Process(target=predict_on_live_audio)
    # p2.start()


if __name__ == "__main__":
    main()
