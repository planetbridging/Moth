import os
import cv2
import pandas as pd
import numpy as np

# Directory containing the images
dir_path = "me"

# List to store the data
data = []

# Loop through the files in the directory
for filename in os.listdir(dir_path):
    # Construct the full file path
    file_path = os.path.join(dir_path, filename)

    # Read the image
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 128x128 pixels
    img_resized = cv2.resize(img, (128, 128))

    # Flatten the image
    img_flattened = img_resized.flatten()

    # Normalize the pixel values to the range [0, 1]
    img_normalized = img_flattened / 255.0

    # Create a row for the CSV file
    # The first two columns are 'MeFound' and 'Looking'
    # The rest of the columns are the pixel values
    #row = [True, "lookingDown"] + img_normalized.tolist()
    row = ['True'] + img_normalized.tolist()

    # Add the row to the data
    data.append(row)

# Column names for the CSV file
columns = ["MeFound"] + [f"Pixel{i}" for i in range(1, len(data[0]))]


# Create a DataFrame from the data
df = pd.DataFrame(data, columns=columns)

# Write the DataFrame to a CSV file
df.to_csv("data2.csv", index=False)
