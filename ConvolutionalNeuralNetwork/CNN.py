#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network

# ### Importing the libraries

# In[8]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[9]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Preprocessing the Training set

# In[10]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# ### Preprocessing the Test set

# In[11]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# ## Part 2 - Building the CNN

# ### Initialising the CNN

# In[14]:


cnn = tf.keras.models.Sequential()


# ### Step 1 - Convolution

# In[15]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# ### Step 2 - Pooling

# In[16]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Adding a second convolutional layer

# In[17]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Step 3 - Flattening

# In[18]:


cnn.add(tf.keras.layers.Flatten())


# ### Step 4 - Full Connection

# In[19]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Step 5 - Output Layer

# In[20]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the CNN

# ### Compiling the CNN

# In[21]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the CNN on the Training set and evaluating it on the Test set

# In[22]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# ## Part 4 - Making a single prediction

# In[27]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
#target_size set the image to the size of 64,64

test_image = image.img_to_array(test_image) 
print(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image)
result = cnn.predict(test_image)
print(training_set.class_indices) #class_indices will give the class name and their respective encoding label.
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'


# In[26]:


print(prediction)


# In[ ]:




