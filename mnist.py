from keras.datasets import mnist

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f'Original data:')
print(f'\tTraining data shape: {x_train.shape}')
print(f'\tTest data shape: {x_test.shape}')

# Flatten the images
# Make from image data some vectors
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
print(f'--------------')

print(f'After vectorisation:')
print(f'\tTraining data shape: {x_train.shape}')
print(f'\tTest data shape: {x_test.shape}')


