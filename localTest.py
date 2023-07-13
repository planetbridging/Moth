import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read a frame from the webcam
ret, frame = cap.read()

# Check if the frame was read correctly
if not ret:
    raise IOError("Cannot read frame")

# Resize the image to 128x128 pixels
img_resized = cv2.resize(frame, (128, 128))

# Convert the image to grayscale
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Normalize the pixel values to the range [0, 1]
#img_normalized = img_gray / 255.0

# Add a batch dimension
#img_batch = np.expand_dims(img_normalized, axis=0)

# Flatten the image
img_flattened = img_gray.flatten()
print(img_flattened)


# Save the grayscale image
cv2.imwrite('photo_gray.jpg', img_gray)

# Save the frame as an image
cv2.imwrite("photo.jpg", frame)


# Define the coordinates of the bounding box
x, y, w, h = 50, 50, 20, 20

# Draw the bounding box
cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save the image
cv2.imwrite('photo_with_box.jpg', img_resized)


# Release the webcam
cap.release()
