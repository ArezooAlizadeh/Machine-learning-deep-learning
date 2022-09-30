# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set

#This one is normalization
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# This one takes images from directory and do rescaling in order to reduce
# computational expenses

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')



# Preprocessing the Test set
# on the testset no ttansformation is applied only feature scaling
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')


# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# Here filter size is filter parameters means how feature detector we want.
# kernel size is size of filter, activation is activation function of colnolutional 
# layer which is rectifier in order to break up any nonlinearity 
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu', input_shape=[64,64,3]))
#filter with size 32 is classical filter


# Step 2 - Pooling
# pool size is the size of pooling matrix
# strides is how many number of pixels is this frame shifter ( this one should 
# be size of matrix since we are going to make maximum of size of matrix)

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu', input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# this layer is hidden layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
# Number of epochs is obtained by try and test to get high accuracy as possible


# Part 4 - Making a single prediction
# you can find those information from https://keras.io/api/preprocessing/image/
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size = (64,64))
# here the image is changed to numpy array
test_image = image.img_to_array(test_image)
# We train and test the model based on epoch so in order to test the model based
# on single predictio, we need to define an extra dimension
# to the image corresponding to the batch. So here test image belongs to the first batch
test_image = np.expand_dims(test_image, axis = 0) 
result = cnn.predict(test_image)
training_set.class_indices # just to see labels of training set
if result[0][0] ==1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)




