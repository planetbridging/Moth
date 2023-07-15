from djitellopy import Tello
import cv2
import time

# Create a Tello object
tello = Tello()

# Connect to the drone
tello.connect()
time.sleep(5)  # Wait for 5 seconds

# Start the video stream
tello.streamon()
time.sleep(2)  # Wait for 2 seconds to let the stream setup

# Create a VideoCapture object
cap = cv2.VideoCapture('udp://@0.0.0.0:11111?overrun_nonfatal=1&fifo_size=50000000')

# Get the next frame from the video stream
ret, frame = cap.read()

# If a frame was received, display it
if ret:
    cv2.imshow('DJI Tello', frame)

# Wait for the user to press a key
cv2.waitKey(0)

# Stop the video stream
tello.streamoff()

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
