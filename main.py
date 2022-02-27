import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

print(tf.__version__)

######################################
######################################
#         Data PreProcessing
######################################
######################################


#preprocessing training set

#apply transformation to avoid overfitting (image augmentation)

train_datagen = ImageDataGenerator(
                rescale = 1/255, #normalising all pixel values between 0 and 1
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

training_set = train_datagen.flow_from_directory(
                '../dataset/training_set',
                target_size = (64,64),
                batch_size = 32,
                class_mode = 'binary')


#preprocessing test set
test_datagen = ImageDataGenerator(rescale=1/255)
test_set = test_datagen.flow_from_directory(
                        '../dataset/test_set',
                        target_size = (64,64),
                        batch_size = 32,
                        class_mode= 'binary')


######################################
######################################
#         Building the CNN
######################################
######################################

#initialise the CNN

cnn = tf.keras.models.Sequential()

#step 1 - Add Convolution layer
cnn.add(tf.keras.layers.Conv2D( filters = 32, kernel_size = 3, activation = 'relu', input_shape = [64,64,3]))

#step 2 - Add max pooling
cnn.add(tf.keras.layers.MaxPool2D( pool_size = 2, strides = 2))

#step 3 - Add another convolutional Layer
cnn.add(tf.keras.layers.Conv2D( filters = 32, kernel_size = 3, activation = 'relu'))

#step 4 = Add another max pooling layer
cnn.add(tf.keras.layers.MaxPool2D( pool_size = 2, strides = 2))


######################################
#         Flattening
cnn.add(tf.keras.layers.Flatten())



######################################
#         Fully Connected Layers
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu' ))


######################################
#         Output layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid' ))



######################################
######################################
#         Training the CNN
######################################
######################################

######################################
#         Compiling CNN
cnn.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )


######################################
#         Training the CNN on the training set and evaluating it on the Test set
cnn.fit( x = training_set, validation_data = test_set, epochs = 25)


######################################
######################################
#         Making a single prediction
######################################
######################################

test_image = image.load_img('../dataset/single_prediction/cat_or_dog_1.jpg', target_size= (64,64))
#convert to np array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image/255)

training_set.class_indices

if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

