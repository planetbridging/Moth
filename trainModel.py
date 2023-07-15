import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv('data2.csv')

# Preprocess the data
# Assuming "Pixel1" to "Pixel16384" are the feature columns
# and "MeFound" is the target column
X = data.loc[:, 'Pixel1':'Pixel16384']
y = data['MeFound']

# Convert the labels to integers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Convert the target variable to float32
y = y.astype('float32')

# Normalize the pixel values to be between 0 and 1
X = X.astype('float32') / 255

# Reshape the data
X = X.values.reshape(-1, 128, 128, 1)  # assuming the images are 128x128 pixels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(BatchNormalization())  # Added Batch Normalization layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))  # Increased dropout rate
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))  # Increased dropout rate
model.add(Dense(1, activation='sigmoid'))  # assuming there are 2 classes

# Compile the model
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Changed optimizer and learning rate
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train,
          batch_size=128,
          epochs=100,
          verbose=1,
          validation_data=(X_test, y_test))

# Evaluate the model on the test data
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model architecture as JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("model_weights.h5")

#export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
