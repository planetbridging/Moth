from djitellopy import Tello
import cv2
import time
import threading

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json


# Load the model architecture from JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model_weights.h5")

# Compile the loaded model
loaded_model.compile(loss=tf.keras.losses.binary_crossentropy,
                     optimizer=tf.keras.optimizers.Adadelta(),
                     metrics=['accuracy'])

# Capture a photo using the webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open the webcam")
    exit()

# Set the width and height of the capture frame
width, height = 128, 128
tryCount = 10

# Create a Tello instance
#tello = Tello()

# Connect to the drone
#tello.connect()
#time.sleep(5)  # Wait for 5 seconds

# Check the battery level
#print("Battery level is: %s" % tello.get_battery())

# Start the video stream
#tello.streamon()

# The drone needs to be in the air to take a picture
#tello.takeoff()

def take_picture():
    # Take a photo
    #frame_read = tello.get_frame_read()
    frame = frame_read.frame

    #cv2.imwrite("./pics/"+f"picture_{time.time()}.jpg", frame)

    # Start another timer
    threading.Timer(5.0, take_picture).start()

# Start the first timer
#take_picture()

def tryCamera():
    # Loop for 10 seconds
    end_time = time.time() + 10
    while time.time() < end_time:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale frame to match the input shape of the model
        resized = cv2.resize(gray, (width, height))

        # Normalize the resized image
        normalized = resized.astype('float32') / 255

        # Reshape the image to match the input shape of the model
        image = np.expand_dims(normalized, axis=-1)
        image = np.expand_dims(image, axis=0)

        # Predict the class probabilities
        probabilities = loaded_model.predict(image)
        print(probabilities)

        # Define the threshold probability for a positive match
        threshold = 0.7

        # Check if the predicted class is a positive match
        if probabilities > threshold:
            print("Match: Found")
        else:
            print("No Match: Not Found")

        # Display the grayscale frame
        cv2.imshow("Frame", gray)

        # Wait for 1 second
        cv2.waitKey(1000)

    # Release the webcam and close the windows
    cap.release()
    cv2.destroyAllWindows()

# Call the tryCamera() function
tryCamera()


    # Display the grayscale frame
    #cv2.imshow("Frame", gray)

    #cv2.imwrite("./pics/"+f"picture_{time.time()}.jpg", frame)

    # Start another timer

tryCamera()

    # Wait for 5 seconds
    #time.sleep(5)

# Land the drone
#tello.land()



# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
