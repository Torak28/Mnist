from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import keras

def evaluate(model, batch_size=128, epochs=5, optimizer='sgd', loss_type='mean_squared_error'):
    model.summary()
    model.compile(optimizer=optimizer, loss=loss_type, metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'model accuracy of {epochs} epochs, {batch_size} batch_size, {optimizer} optimizer and {loss_type} loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'Original data:')
print(f'\tTraining data shape: {x_train.shape}')
print(f'\tTest data shape: {x_test.shape}')

# Flatten the images
# Make from image data some vectors
# And do some normalization
image_vector_size = 28*28

x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape(x_test.shape[0], image_vector_size)
x_test = x_test.astype('float32')
x_test /= 255

print(f'--------------')

print(f'After vectorisation:')
print(f'\tTraining data shape: {x_train.shape}')
print(f'\tTest data shape: {x_test.shape}')

print(f'--------------')
print(f'Num of train examples: {x_train.shape[0]}')
print(f'Num of test examples: {x_test.shape[0]}')

# One-hot encoded vector
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f'--------------')
print(f'One- hot encoded vector: {y_train[0]}')

model = Sequential()

model.add(Dense(units=32, activation='sigmoid', input_shape=(image_vector_size,)))
model.add(Dense(units=num_classes, activation='softmax'))

evaluate(model, batch_size=128, epochs=5, optimizer='sgd', loss_type='mean_squared_error')
evaluate(model, batch_size=128, epochs=5, optimizer='sgd', loss_type='categorical_crossentropy')
evaluate(model, batch_size=128, epochs=5, optimizer='adam', loss_type='categorical_crossentropy')
