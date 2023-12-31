from djitellopy import Tello
import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import model_from_json


# Load the model architecture from JSON file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model_weights.h5")

# Compile the loaded model
loaded_model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adadelta(),
    metrics=['accuracy']
)

# Create a Tello instance
tello = Tello()

# Connect to the drone
tello.connect()
time.sleep(5)  # Wait for 5 seconds
tello.streamon()

# Set the width and height of the capture frame
width, height = 320, 240

# Create an OpenCV window
cv2.namedWindow("Drone View", cv2.WINDOW_NORMAL)

while True:
    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read a frame from the video stream of the drone
    frame = tello.get_frame_read().frame

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

    # Display the grayscale frame in the window
    cv2.imshow("Drone View", gray)

# Land the drone and end the video stream
tello.streamoff()
# tello.land()

# Release the window
cv2.destroyAllWindows()
