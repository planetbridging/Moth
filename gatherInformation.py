import cv2
import time

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Start time
start_time = time.time()

# Number of images to capture
#incase everything is root
#chmod -R a+rwx .

num_images = 100

# Image counter
img_counter = 0

while img_counter < num_images:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read correctly
    if not ret:
        raise IOError("Cannot read frame")

    # Resize the image to 128x128 pixels
    img_resized = cv2.resize(frame, (128, 128))

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite("me/"+f'photo_gray_{img_counter}.jpg', img_gray)

    # Save the frame as an image
    #cv2.imwrite("me/"+f"photo_{img_counter}.jpg", frame)

    # Increment image counter
    img_counter += 1

    # Wait for 1 second before capturing the next image
    time.sleep(1)

# Release the webcam
cap.release()
