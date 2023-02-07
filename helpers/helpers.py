from tensorflow import keras
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from .constants import Config
import librosa


def _create_live_video_window():
    cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Video", 800, 600)


def _create_rolling_average_window(window_size):
    return deque(maxlen=window_size)


def _load_model(model_path):
    return keras.models.load_model(model_path)


def _open_video_file(file_path):
    video_reader = cv2.VideoCapture(file_path)
    video_reader.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    video_reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    return video_reader


def _open_live_webcam():
    video_reader = cv2.VideoCapture(Config.STREAMING_MODE)
    video_reader.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_reader.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return video_reader


def _create_video_writer(video_reader, media_path):
    file_name = f'video-{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}.mp4'
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(
        f"{media_path}/{file_name}",
        cv2.VideoWriter_fourcc("M", "P", "4", "V"),
        24,
        (original_video_width, original_video_height),
    )
    return video_writer, file_name


def _process_and_predict(model, audio_data):
    """
    Processes audio data and returns the model's predictions.

    Args:
        model: YAMNet model object.
        audio_data: numpy array of shape (num_samples,) containing audio samples.


    Returns:
        scores: numpy array of shape (batch_size, num_classes) containing the scores for each class.
        melspec: numpy array of shape (batch_size, time_steps, num_mels) containing the mel spectrogram
            of the audio.
    """
    # Convert audio data from int16 to float32
    audio_data = librosa.util.buf_to_float(audio_data, n_bytes=2, dtype=np.int16)

    # Reshape audio data for model input
    audio_data = np.reshape(audio_data, (1, -1))

    # Predict scores and mel spectrogram for the audio
    scores, melspec = model.predict(audio_data, steps=1)

    return scores, melspec
