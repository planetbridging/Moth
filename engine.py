from djitellopy import Tello
import cv2
import time
import threading

# Create a Tello instance
tello = Tello()

# Connect to the drone
tello.connect()
time.sleep(5)  # Wait for 5 seconds

# Check the battery level
print("Battery level is: %s" % tello.get_battery())

# Start the video stream
tello.streamon()

# The drone needs to be in the air to take a picture
#tello.takeoff()

def take_picture():
    # Take a photo
    frame_read = tello.get_frame_read()
    frame = frame_read.frame

    #cv2.imshow('DJI Tello', frame)
    cv2.imwrite("./pics/"+f"picture_{time.time()}.jpg", frame)
    #cv2.imshow('DJI Tello', frame)
    #time.sleep(5)  # Wait for 5 seconds

    # Start another timer
    threading.Timer(5.0, take_picture).start()

# Start the first timer
take_picture()


# Close all OpenCV windows
cv2.destroyAllWindows()

    # Wait for 5 seconds
    #time.sleep(5)

# Land the drone
#tello.land()
