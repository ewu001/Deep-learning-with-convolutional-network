from model import residual_model 
import util as util

import keras

res_model = residual_model.ResNet_50(input_shape=(64, 64, 3), classes=6, zero_padding=3)

res_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = util.load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = keras.utils.to_categorical(Y_train_orig.reshape(-1))
Y_test = keras.utils.to_categorical(Y_test_orig.reshape(-1))

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

res_model.fit(X_train, Y_train, epochs = 2, batch_size = 32)
prediction = res_model.evaluate(X_test, Y_test)
print ("Loss = " + str(prediction[0]))
print ("Test Accuracy = " + str(prediction[1]))
