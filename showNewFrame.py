import cv2
import time
import os
import glob

# Function to get the latest file in a directory
def get_latest_file(path):
    list_of_files = glob.glob(path + '/*')
    if not list_of_files:  # I.e. if the list is empty.
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# Function to display the latest image in a directory
def display_latest_image(path):
    while True:
        latest_file = get_latest_file(path)
        if latest_file is not None:
            img = cv2.imread(latest_file)
            cv2.imshow('Latest Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)  # Update the image every second

# Display the latest image in the 'pics' directory
display_latest_image('pics')

# Close all OpenCV windows
cv2.destroyAllWindows()
