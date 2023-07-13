import cv2
import tensorflow as tf

# Load the trained model
model = tf.saved_model.load('path_to_your_model')

# Load an image
img = cv2.imread('image.jpg')

# Convert the image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to a tensor
input_tensor = tf.convert_to_tensor(img_rgb)
input_tensor = input_tensor[tf.newaxis,...]

# Run the model
output_dict = model(input_tensor)

# Draw the bounding boxes on the image
for i in range(len(output_dict['detection_scores'])):
    if output_dict['detection_scores'][i] > 0.5:
        y_min, x_min, y_max, x_max = output_dict['detection_boxes'][i]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Save the image
cv2.imwrite('image_with_boxes.jpg', img)
