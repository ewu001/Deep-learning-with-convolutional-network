# Import modules
# Add modules as needed
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from model import keras_model

# Configure the model mode
# Available options: CNN, DNN
# Configure hyper parameters
MODEL_MODE = "DNN"
DNN_EPOCH = 12
CNN_EPOCH = 3
OPTIMIZER = 'Adam'
BATCH_SIZE = 128

# Import MNIST dataset from openml
scikit_learn_data_path = './scikit_learn_data'
dataset = fetch_openml('mnist_784', version=1, data_home=scikit_learn_data_path)

# Data pre-process and normalization
data_X = dataset['data']
data_X = data_X.astype('float32')
data_X /= 255
data_Y = dataset['target']
print(data_X.shape)

# Apply one-hot encoding to data_Y
one_hot_Y = keras.utils.to_categorical(data_Y)



# Split data into a train set (50%), validation set (20%) and a test set (30%)
X_train, X_remain, y_train, y_remain = train_test_split(data_X, one_hot_Y, test_size=0.5, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_remain, y_remain, test_size=0.6, random_state=42)

print(X_train.shape)
print(X_validation.shape)
print(X_test.shape)


# Build your neural network structure
if MODEL_MODE == "CNN":
    md = keras_model.CNNmodel()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    EPOCH = CNN_EPOCH
else:
    md = keras_model.DNNmodel()  
    EPOCH = DNN_EPOCH   

# Print model summary for architecture and parameter viewing
md.summary()


# Compile your model
md.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping:
stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train model
history = md.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(X_validation, y_validation), callbacks=[stopping])
print("Finished training")


# Output the loss and accuracy on the test set
print("Evaluate model on test set: ")
test_loss, test_acc = md.evaluate(x=X_test, y=y_test)
print("Test set accuracy: ", test_acc)


# Optional, like really optional, only do this if you're done with training your model
# Make a tensorboard callback object to see the results live in tensorboard!
#
# Can use this code
# tb_callback = keras.callbacks.TensorBoard(log_dir='some path', write_graph=True, write_images=True)
#
# You will also need to start tensorboard from another cmd window
# In another cmd window in the virtual environment, start tensorboard with: tensorboard --logdir somePathYouDefinedInCallback
# This will give you a website that you can visit on your browser