import os
import cv2
import math
import pafy
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt
from tensorflow import keras
import youtube_dl

"""
    - Introduction to Video Classification and Human Activity Recognition:
        * https://learnopencv.com/introduction-to-video-classification-and-human-activity-recognition/

"""
# matplotlib inline

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense


import sys
import warnings

import winerror
import win32api
import win32job

g_hjob = None


IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
MAX_IMAGES_PER_CLASS = 8000

DATASET_DIRECTORY = "UCF_CRIME"
CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

CLASSES_LIST = os.listdir(DATASET_DIRECTORY)  # Return all classes name in UCF50


# CLASSES_LIST = ["HorseRace"]
MODEL_OUTPUT_SIZE = len(CLASSES_LIST)


"""
    Memory section 
"""


def data_demo():
    # Create a Matplotlib figure
    plt.figure(figsize=(30, 30))

    # Get Names of all classes in UCF50
    all_classes_names = os.listdir("UCF50")

    # Generate a random sample of images each time the cell runs
    random_range = random.sample(range(len(all_classes_names)), 20)

    # Iterating through all the random samples
    for counter, random_index in enumerate(random_range, 1):

        # Getting Class Name using Random Index
        selected_class_Name = all_classes_names[random_index]

        # Getting a list of all the video files present in a Class Directory
        video_files_names_list = os.listdir(f"UCF50/{selected_class_Name}")

        # Randomly selecting a video file
        selected_video_file_name = random.choice(video_files_names_list)

        # Reading the Video File Using the Video Capture
        video_reader = cv2.VideoCapture(
            f"UCF50/{selected_class_Name}/{selected_video_file_name}"
        )

        # Reading The First Frame of the Video File
        _, bgr_frame = video_reader.read()

        # Closing the VideoCapture object and releasing all resources.
        video_reader.release()

        # Converting the BGR Frame to RGB Frame
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Adding The Class Name Text on top of the Video Frame.
        cv2.putText(
            rgb_frame,
            selected_class_Name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        # Assigning the Frame to a specific position of a subplot
        plt.subplot(5, 4, counter)
        plt.imshow(rgb_frame)
        plt.axis("off")


def frames_extraction(video_path):
    """
    -   A function that will extract frames from each video while performing other
        preprocessing operation like resizing and normalizing images.

    -   Takes a video file path as input. It then reads the video file frame by frame,
        resizes each frame, normalizes the resized frame, appends the normalized frame into a list,
        and then finally returns that list.
    """
    # Empty List declared to store video frames
    frames_list = []

    # Reading the Video File Using the VideoCapture
    video_reader = cv2.VideoCapture(video_path)

    # Iterating through Video Frames
    while True:

        # Reading a frame from the video file
        success, frame = video_reader.read()

        # If Video frame was not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Appending the normalized frame into the frames list
        frames_list.append(normalized_frame)

    # Closing the VideoCapture object and releasing all resources.
    video_reader.release()

    # returning the frames list
    return frames_list


def create_dataset():

    # Declaring Empty Lists to store the features and labels values.
    temp_features = []
    features = []
    labels = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f"Extracting Data of Class: {class_name}")

        # Getting the list of video files present in the specific class name directory
        files_list = os.listdir(os.path.join(DATASET_DIRECTORY, class_name))

        # Iterating through all the files present in the files list
        for file_name in files_list:

            # Construct the complete video path
            video_file_path = os.path.join(DATASET_DIRECTORY, class_name, file_name)

            # Calling the frame_extraction method for every video file path
            frames = frames_extraction(video_file_path)

            # Appending the frames to a temporary list.
            temp_features.extend(frames)

        # Adding randomly selected frames to the features list
        features.extend(random.sample(temp_features, MAX_IMAGES_PER_CLASS))

        # Adding Fixed number of labels to the labels list
        labels.extend([class_index] * MAX_IMAGES_PER_CLASS)

        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()

    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels


# Let's create a function that will construct our model
def create_model():

    # We will use a Sequential model for model construction
    model = Sequential()

    # Defining The Model Architecture
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        )
    )
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(MODEL_OUTPUT_SIZE, activation="softmax"))

    # Printing the models summary
    model.summary()

    return model


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    # Get Metric values using metric names as identifiers
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Constructing a range object which will be used as time
    epochs = range(len(metric_value_1))

    # Plotting the Graph
    plt.plot(epochs, metric_value_1, "blue", label=metric_name_1)
    plt.plot(epochs, metric_value_2, "red", label=metric_name_2)

    # Adding title to the plot
    plt.title(str(plot_name))

    # Adding legend to the plot
    plt.legend()
    plt.show()


def start_model_training():
    ## **Step 2: Visualize the Data with its Labels**
    # data_demo()
    # Set Numpy, Python and Tensorflow seeds to get consistent results.
    print("Starting model training...")
    seed_constant = 23
    np.random.seed(seed_constant)
    random.seed(seed_constant)
    tf.random.set_seed(seed_constant)
    # ----------------------------------------
    ## **Step 3: Read & Preprocess the Dataset**
    # IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
    # MAX_IMAGES_PER_CLASS = 8000

    # DATASET_DIRECTORY = "UCF50"
    # CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
    # MODEL_OUTPUT_SIZE = len(CLASSES_LIST)

    # ----------------------------------------
    ### **Dataset Creation**
    ## Calling the **create_dataset** method which returns features and labels.
    features, labels = create_dataset()
    # Using Keras's to_categorical method to convert labels into one-hot-encoded vectors
    one_hot_encoded_labels = to_categorical(labels)
    ## **Step 4: Split the Data into Train and Test Set**
    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        one_hot_encoded_labels,
        test_size=0.2,
        shuffle=True,
        random_state=seed_constant,
    )
    ## **Step 5: Construct the Model**
    # Calling the create_model method
    model = create_model()
    print("Model Created Successfully!")

    ### **Check Model’s Structure:**
    plot_model(
        model,
        to_file="model_structure_plot.png",
        show_shapes=True,
        show_layer_names=True,
    )

    ## **Step 6: Compile & Train the Model**
    # Adding the Early Stopping Callback to the model which will continuously monitor the validation loss metric for every epoch.
    # If the models validation loss does not decrease after 15 consecutive epochs, the training will be stopped and the weight which reported the lowest validation loss will be retored in the model.
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=15, mode="min", restore_best_weights=True
    )

    # Adding loss, optimizer and metrics values to the model.
    model.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )

    # Start Training
    model_training_history = model.fit(
        x=features_train,
        y=labels_train,
        epochs=50,
        batch_size=4,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping_callback],
    )

    ### **Evaluating Your Trained Model**

    model_evaluation_history = model.evaluate(features_test, labels_test)
    # Creating a useful name for our model, incase you're saving multiple models (OPTIONAL)

    date_time_format = "%Y_%m_%d__%H_%M_%S"
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(
        current_date_time_dt, date_time_format
    )
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    model_name = f"Model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5"

    # Saving your Model
    model = model.save(model_name)
    ## **Step 7: Plot Model’s Loss & Accuracy Curves**
    plot_metric(
        model_training_history,
        "accuracy",
        "val_accuracy",
        "Total Accuracy vs Total Validation Accuracy",
    )
    plot_metric(
        model_training_history,
        "loss",
        "val_loss",
        "Total Loss vs Total Validation Loss",
    )


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
    model, video_file_path=None, output_file_path=None, window_size=25
):
    cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Video", 800, 600)

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen=window_size)

    # # Reading the Video File using the VideoCapture Object
    # video_reader = cv2.VideoCapture(video_file_path)

    # # Getting the width and height of the video
    # original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    # original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoCapture object to read from the default webcam
    video_reader = cv2.VideoCapture(0)
    # Gettting video FPS
    # fps = video_reader.get(cv2.CAP_PROP_FPS)
    # fps = 30
    # video_reader.set(cv2.CAP_PROP_FPS, fps)
    # Getting the width and height of the webcam frames
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


def call_on_video():
    # Creating The Output directories if it does not exist
    print("Creating The Output directories if it does not exist  ...")
    output_directory = "Youtube_Videos"
    os.makedirs(output_directory, exist_ok=True)

    print("Downloading a YouTube Video  ...")

    video_url = "https://www.youtube.com/watch?v=CX1jv1ZLxGg"
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
    # Only use if training a new model is  required
    # start_model_training()
    print("Loading model classes", CLASSES_LIST)
    print("Loading model  ...")
    model = keras.models.load_model(
        "Model___Date_Time_2023_01_15__12_21_55___Loss_0.3903857469558716___Accuracy_0.9278905987739563.h5"
    )

    # Uncomment to pass variables to 'predict_on_live_video' then in 'predict_on_live_video' change to non-live also
    # output_video_file_path, input_video_file_path, window_size = call_on_video()
    # # Calling the predict_on_live_video method to start the Prediction.
    # predict_on_live_video(
    #     model, input_video_file_path, output_video_file_path, window_size
    # )

    print("Calling the predict_on_live_video method to start the Prediction.")
    # # Calling the predict_on_live_video method to start the Prediction.
    predict_on_live_video(model)

    # Play Video File in the Notebook
    # VideoFileClip(output_video_file_path).ipython_display(width = 700)


if __name__ == "__main__":
    main()
