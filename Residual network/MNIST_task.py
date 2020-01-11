import keras
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


from model import residual_model 

BATCH_SIZE = 128
EPOCH = 1

# Test on MNIST data set
# Import MNIST dataset from openml
scikit_learn_data_path = 'data/scikit_learn_data'
dataset = fetch_openml('mnist_784', version=1, data_home=scikit_learn_data_path)

# Data pre-process and normalization
data_X = dataset['data']
data_X = data_X.astype('float32')
data_X /= 255
data_Y = dataset['target']
print(data_X.shape)


# Apply one-hot encoding to data_Y
one_hot_Y = keras.utils.to_categorical(data_Y)

#Split data into a train set (50%), validation set (20%) and a test set (30%)
X_train, X_remain, y_train, y_remain = train_test_split(data_X, one_hot_Y, test_size=0.5, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_remain, y_remain, test_size=0.6, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# Create instance of the residual model and compile
res_model = residual_model.ResNet_50(input_shape=(28, 28, 1), classes=10, zero_padding=21)
res_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping:
stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train model
history = res_model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(X_validation, y_validation), callbacks=[stopping])
print("Finished training")


# Output the loss and accuracy on the test set
print("Evaluate model on test set: ")
test_loss, test_acc = res_model.evaluate(x=X_test, y=y_test)
print("Test set accuracy: ", test_acc)
