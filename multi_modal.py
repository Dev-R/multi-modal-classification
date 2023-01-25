# importing libraries for audio classification
import time
import threading
import cv2
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

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
from helpers.constants import ActionRecognition, AudioRecognition, ObjectDetection, Config


NOTIFICATION_ON_GOING = False
ANOMALY_DICT = {
    "video_anomaly": {"is_anomaly": False, "name": None, "video_clip_url": None},
    "audio_anomaly": {"is_anomaly": False, "name": "N/A"},
    "SISR": {"frame_exists": False, "enhanced_frame_url": ""},
    "SMS": {"phone_number": None, "message": None},
}


"""
    TODO
    # Morning/night mode?(based on time???)
    # if morning mode:
    #         -> video will be the main classifiers
    #         -> audio will be sent also
    # if night mode:
    #         -> audio will be the main classifiers
    #         -> only audio will be sent


    If video detect anomaly detected:
        1- Trigger possible anomaly ✔
        2- Save anomaly frame and enhance with ESRGAN ✔
        3- Save next 5 video seconds ✔
        4.
            4.1 - Upload frame to blob ✔
            4.2 - Upload video to blob ✔
        5- Trigger SMS services:
            5.1 - Send SMS Message to owner containing:
                5.1.1 - Message text/ data-classificaiton?  ✔
                5.1.2 - Message media ✔
                5.1.3 - Date and other meta info ✔


"""


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

    if not NOTIFICATION_ON_GOING:
        is_image_uploaded = False  # Will be true if img is uploaded to cloud
        is_video_uploaded = False  # Will be true if video is uploaded to cloud
        NOTIFICATION_ON_GOING = True
        print("Possible red flag 🚩:", anomaly_name)
        img_path_name = f"{Config.MEDIA_PATH}/frame.png"
        # convert frame to image
        cv2.imwrite(img_path_name, frame)
        # If video frame is provided
        if frame.any():
            try:
                print("Frame provided, Calling ESRGAN API ...")
                img_blob_url = apply_esragan(img_path_name)
                ANOMALY_DICT["SISR"]["frame_exists"] = True
                ANOMALY_DICT["SISR"]["enhanced_frame_url"] = img_blob_url
                print("Image file uploaded ✔")
                is_image_uploaded = True
            except Exception as ex:
                print("Unable to upload image file to Azure BLOB ❌:", ex)
        # If video file is provided
        if video:
            try:
                print("Uploading video file to Azure BLOB ...")
                video_blob_url = upload_blob(video_name, is_video=True)
                ANOMALY_DICT["video_anomaly"]["is_anomaly"] = True
                ANOMALY_DICT["video_anomaly"]["name"] = anomaly_name
                ANOMALY_DICT["video_anomaly"]["video_clip_url"] = video_blob_url
                print("Video file uploaded ✔")
                is_video_uploaded = True
            except Exception as ex:
                print("Unable to upload video file to Azure BLOB ❌:", ex)
        # If both video and image are uploaded, notify owner
        if is_video_uploaded and is_image_uploaded:
            print("Sending notification to owner 🔔 ...")
            notify_owner(
                notification_data={
                    "type": Config.NOTIFICATION_CONFIG["notification_type"],
                    "data": {
                        "location": "KL",
                        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "audio_classification": ANOMALY_DICT["audio_anomaly"]["name"],
                        "video_classification": ANOMALY_DICT["video_anomaly"]["name"],
                        "image_link": img_blob_url,
                        "video_link": video_blob_url,
                    },
                }
            )

    else:
        pass


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
    print("Notification daemon on going...")
    tqdm(time.sleep(Config.NOTIFICATION_CONFIG["notification_interval"]))
    NOTIFICATION_ON_GOING = False
    print("Notification daemon stopped", Config.NOTIFICATION_CONFIG)


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
    print("predict_on_live_audio")
    # Load YAMNet model and class names
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("audio_classification_YAMNET/yamnet/yamnet.h5")
    class_names = yamnet_model.class_names(
        "audio_classification_YAMNET/yamnet/yamnet_class_map.csv"
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
        scores, melspec =  helpers._process_and_predict(yamnet, data)
        scores_np = np.array(scores)
        mean_scores = np.mean(scores, axis=0)
        # print('mean_scores = np.mean(scores, axis=0)', mean_scores)
        top_class_indices = np.argsort(mean_scores)[::-1][:TOP_N]

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
        video_reader = helpers._open_live_webcam()

    video_writer, video_file_name = helpers._create_video_writer(video_reader, Config.MEDIA_PATH)

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
        ObjectDetection.PROTOTXT,
        ObjectDetection.MODEL,
    )

    print("[INFO] starting video stream...")
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
            font_size = 0.55

            sizes = [font_size for _ in range(len(top_5_class_names))]
            for i, class_name in enumerate(top_5_class_names):
                class_confidence = round(
                    predicted_labels_probabilities_averaged[top_5_indices[i]], 3
                )
                class_name = f"{class_name}  (Confidence= {str(class_confidence)})"
                current_top_class_name = top_5_class_names[0]
                # Define the positions for each class name
                positions = [
                    (10 * (i + 1), 30),
                    (160 * (i + 1), 30),
                    (10 * (i + 1), 400),
                    (90 * (i + 1), 400),
                    (2000 * (i + 1), 400),
                ]
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

                if (
                    current_top_class_name in ActionRecognition.ANOMALY_CLASSES_NAME
                    and not NOTIFICATION_ON_GOING
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
                print("Object detected: ", label)
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


if __name__ == "__main__":
    # func1()
    # predict_on_live_audio()
    # predict_on_live_video()
    # # predict_on_live_audio_and_video()
    p1 = Process(target=predict_on_live_video)
    p1.start()
    # p2 = Process(target=predict_on_live_audio)
    # p2.start()
