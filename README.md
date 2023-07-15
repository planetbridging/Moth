# DJI Tello Object Recognition and Autonomous Flight

This project aims to enable a DJI Tello drone to recognize a specific person and fly towards them using TensorFlow and Python on Kali Linux Purple.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The project utilizes TensorFlow and Python to enable the DJI Tello drone to perform object recognition of a specific person and autonomously fly towards them. This can be achieved by training a deep learning model on a dataset of images, implementing the recognition algorithm, and integrating it with the Tello drone control system.

## Installation

To install and set up the project, follow these steps:

1. Install Kali Linux Purple on your system. [Instructions](https://www.kali.org/docs/general-use/install-nvidia-drivers-on-kali-linux/)
2. Install the required dependencies, such as TensorFlow, OpenCV, and the Tello SDK.
3. Clone this repository to your local machine.
4. Follow the instructions in the [Usage](#usage) section to configure and run the project.

## Usage

To use the project, follow these steps:

1. Connect your DJI Tello drone to your Kali Linux Purple system.
2. Prepare the dataset of images for training the recognition model.
3. Train the deep learning model using TensorFlow on the prepared dataset.
4. Update the model and recognition code with the trained weights and algorithms.
5. Execute the project code to start the recognition and autonomous flight.

## Dataset

The dataset used for training the recognition model should consist of images containing the specific person the drone needs to recognize. Collect images with different poses, angles, and lighting conditions to ensure the model's robustness. Annotate the images with bounding boxes or labels to indicate the presence of the person. Unfortunately, we cannot provide a specific dataset for this project, but you can create your own using images of the target person.

## Model Training

The recognition model is trained using TensorFlow on the dataset prepared for the project. Follow these steps to train the model:

1. Preprocess the dataset by resizing the images, normalizing the pixel values, and splitting it into training and validation sets.
2. Design and configure the deep learning model architecture, considering convolutional layers, pooling layers, and fully connected layers.
3. Define appropriate loss function, optimizer, and evaluation metrics for the model.
4. Train the model using the prepared dataset and evaluate its performance on the validation set.
5. Save the trained model weights for later use in the recognition and flight control code.

## Model Inference

After training the model, it can be used for inference to recognize the target person and control the drone accordingly. Follow these steps to perform inference:

1. Load the trained model weights into the recognition code.
2. Capture live video feed from the Tello drone or a connected camera.
3. Process the video frames by resizing, normalizing, and feeding them into the trained model.
4. Analyze the model's predictions to detect the presence of the target person.
5. Control the Tello drone to fly autonomously towards the recognized person.

## Results

Document and share the results of your project here. Include information on the accuracy of the recognition model, the performance of the autonomous flight algorithm, and any other relevant metrics or observations. Provide visual examples or videos to showcase the project's capabilities.

## Contributing

Contributions to this project are welcome! If you encounter any issues, have suggestions, or want to contribute new features, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Submit a pull request, describing your changes and the problem they solve.

## To train your own model

1. Run python3 trainMode.py - this trains the model and saves it to json and h5 format to load and test it
2. Run python3 convertImgToCsv.py to convert the images to csv data to prepare the data for the model
3. Run python3 bulkUpdate.csv and adjust the array for when the person/thing is on the image it says true and false if it isnt which then updates the data in the csv
4. Run Python3 modelTest.py to check how well the model works on your computer and webcam I got 50%ish which sometimes recognize if I'm on or off the screen. Model could be improved
5. All my data and models are in this link https://drive.google.com/drive/folders/1nYh4WooknSNQGILEzXuDFIYp6qeYELea?usp=sharing
