import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

DROPOUT_RATE = 0.25

def DNNmodel():
    # Deep neural network using Keras sequential API.
    # Use ReLu as activation, use Dropout

    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784, )))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(10, activation='softmax'))
    return model

def CNNmodel():
    model = Sequential()
    # First feature extraction block
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                 activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                 activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(DROPOUT_RATE))

    # Second Feature Extraction block
    model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 
                 activation ='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 
                 activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(DROPOUT_RATE))

    # To FC NN block, use dropout
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(DROPOUT_RATE * 2))
    model.add(Dense(10, activation="softmax"))
    return model