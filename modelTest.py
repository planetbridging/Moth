import cv2
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
loaded_model.compile(loss=tf.keras.losses.categorical_crossentropy,
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

while True:
    # Wait for a key press to capture and process a new photo
    cv2.waitKey(0)

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
    probabilities = loaded_model.predict(image)[0]
    print(probabilities)
    # Get the predicted class label
    predicted_class = np.argmax(probabilities)

    # Define the threshold probability for a positive match
    threshold = 0.5

    # Check if the predicted class is a positive match
    if probabilities[predicted_class] > threshold:
        print("Match: Found")
    else:
        print("No Match: Not Found")

    # Display the grayscale frame
    cv2.imshow("Frame", gray)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
