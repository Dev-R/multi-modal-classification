"""
Classifies sounds from a live audio stream using the YAMNet model.

The YAMNet model is a deep convolutional neural network trained to recognize a wide range
of sounds from human speech to animal calls to musical instruments. It has been trained
on a diverse dataset of over 2 million YouTube audio clips and can recognize over 5,000
unique sound classes.

This script captures audio from the default microphone, processes it in real-time, and
visualizes the mel spectrogram of the audio. It also prints out the top 5 predicted
sound classes and their corresponding scores.

Note: This script requires PyAudio, librosa, and TensorFlow to be installed.

'''
    - https://lnu.diva-portal.org/smash/get/diva2:1605037/FULLTEXT01.pdf:
    - https://github.com/SangwonSUH/realtime_YAMNET
    - https://www.tensorflow.org/hub/tutorials/yamnet
    - https://github.com/daisukelab/ml-sound-classifier
'''

"""

import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import keras

import yamnet.params as params
import yamnet.yamnet as yamnet_model

# importing library
import numpy


def get_top_predictions(scores, class_names, top_n=5):
    """Gets the top top_n predictions and their corresponding scores.
    Args:
        scores: numpy array of shape (batch_size, num_classes) containing the scores for each class.
        class_names: list of length `num_classes` containing the names of each class.
        top_n: integer, the number of top predictions to return.

    Returns:
        top_predictions: list of tuples of the form (class_name, score) for the top predictions.
    """
    top_predictions = []
    scores = np.mean(scores, axis=0)
    top_n_indices = np.argsort(scores)[::-1][:top_n]
    for i in top_n_indices:
        top_predictions.append((class_names[i], scores[i]))
    return top_predictions, top_n_indices


def process_and_predict(model, audio_data):
    """Processes audio data and returns the model's predictions.
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


def main():
    # Load YAMNet model and class names
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights("yamnet/yamnet.h5")
    class_names = yamnet_model.class_names("yamnet/yamnet_class_map.csv")
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
    top_n = 10
    plt.ion()
    plt.figure(figsize=(10, 6))
    while True:
        # Read audio data from the microphone
        data = stream.read(frame_len, exception_on_overflow=False)

        # Process audio data and get predictions from the model
        scores, melspec = process_and_predict(yamnet, data)
        scores_np = numpy.array(scores)
        mean_scores = np.mean(scores, axis=0)
        # print('mean_scores = np.mean(scores, axis=0)', mean_scores)
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]

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
        # Label the top_N classes.
        yticks = range(0, top_n, 1)
        plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])


        plt.pause(0.001)
        plt.show()

        # display_image(top_predictions)
        # Print the top predictions and current count
        # print("Current event:")
        # for class_name, score in top_predictions:
        #     print(f"  {class_name}: {score:.3f}")
        # print(f"Count: {cnt}")
        # cnt += 1

    # Clean up PyAudio resources
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
