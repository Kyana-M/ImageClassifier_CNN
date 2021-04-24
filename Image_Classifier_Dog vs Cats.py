#!/usr/bin/env python
# coding: utf-8

# In[190]:


#Essential Imports

import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.utils import shuffle


# In[194]:



# Function to load Image 
def load_image(file_path):    
    return cv2.imread(file_path)
    
# Function to extract label from image    
def extract_label(file_name):
    return 1 if "dog" in file_name else 0


# defining the directory for the training dataset
train_path  =  r'C:\Users\isi\Desktop\Deep Learning Exercise\Image Classifier\train_set'
image_files = os.listdir(train_path)  #copy all the file in this image_files list

#Iteration over the image_files to extract actual images and the corresponding labels 
train_labels = [extract_label(train_path + file) for file in image_files ]
train_images =[load_image(os.path.join(train_path,file)) for file in image_files]


    
#shuffling two list correspondingly    



train_images , train_labels = shuffle(train_images , train_labels)

    


# In[195]:


# preprocessing Image : choosing minimum dimensio from image then extracting the square shape from image and resizing the image
# and converting the image to grayscale from black and white

def preprocess_img(img):
    min_side = min(img.shape[0] , img.shape[1])
    img  = img[:min_side , :min_side]
    img = cv2.resize(img , (100,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img / 255.0
    


# In[196]:


import matplotlib.pyplot as plt
preview_index = 50
plt.subplot(1,2,1)
plt.imshow(train_images[preview_index])
plt.subplot(1,2,2)
plt.imshow(preprocess_img(train_images[preview_index]) , cmap="gray")


# In[197]:


#iterating the over images
for image in range(len(train_images)):
    train_images[image] = preprocess_img(train_images[image])


# In[198]:


print(train_images[1])


# In[220]:


#Converting the train_images list to array
train_images = np.array(train_images)


# In[221]:


print(train_images.shape)


# In[222]:


#expandig the dimension in order to input in cnn adding extra dimension making into 60 * 60 * 1 

train_images = np.expand_dims(train_images , axis = -1)


# In[223]:


train_labels = np.array(train_labels)


# In[224]:


print(train_labels)


# In[225]:


print(train_labels.shape)
print(train_images.shape[1:])


# In[226]:


# defining the cnn model 
# Always input shape for the first layer 60 * 60 * 1


# In[227]:


layers = [
    tf.keras.layers.Conv2D(filters = 16 , kernel_size = (3,3) , padding = "same" , activation = tf.nn.relu , input_shape = train_images.shape[1:]),
    tf.keras.layers.MaxPool2D(pool_size = (2,2) , strides =(2,2)),
    tf.keras.layers.Conv2D(filters = 32 , kernel_size = (3,3) , padding = "same" , activation = tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size = (2,2) , strides =(2,2)),
    tf.keras.layers.Conv2D(filters = 64 , kernel_size = (3,3) , padding = "same" , activation = tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size = (2,2) , strides =(2,2)),
    tf.keras.layers.Conv2D(filters = 128 , kernel_size = (3,3) , padding = "same" , activation = tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size = (2,2) , strides =(2,2)),
    tf.keras.layers.Conv2D(filters = 256 , kernel_size = (3,3) , padding = "same" , activation = tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size = (2,2) , strides =(2,2)),
    tf.keras.layers.Conv2D(filters = 512 , kernel_size = (3,3) , padding = "same" , activation = tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size = (2,2) , strides =(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 512 , activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 256 , activation = tf.nn.relu),
    tf.keras.layers.Dense(units = 2 , activation = tf.nn.softmax)
    
]

model = tf.keras.Sequential(layers)
model.compile(optimizer = tf.optimizers.Adam(),
             loss = tf.losses.SparseCategoricalCrossentropy(),
             metrics = [tf.metrics.SparseCategoricalAccuracy()])


# In[207]:


model.fit(train_images, train_labels ,epochs = 5 ,batch_size = 50)


# In[208]:


model.save_weights("model_weights.tf")


# In[209]:


test_path  =  r'C:\Users\isi\Desktop\Deep Learning Exercise\Image Classifier\test_set'
image_files = os.listdir(test_path)

test_images =[preprocess_img(load_image(os.path.join(test_path,file))) for file in image_files]






# In[210]:


print(test_images)


# In[211]:



eval_model =  tf.keras.Sequential(layers)
eval_model.load_weights("model_weights.tf")


# In[212]:


test_images = shuffle(test_images)


# In[213]:


predictions = eval_model.predict(np.expand_dims(test_images , axis = -1))


# In[214]:


print(predictions.shape)


# In[217]:


cols = 4
rows = np.ceil(len(test_images)/cols)
fig = plt.gcf()
fig.set_size_inches(cols *4 , rows *4)


# In[219]:


for i in range(0,len(test_images)):
    plt.subplot(rows, cols , i+1)
    plt.imshow(test_images[i] , cmap ="gray")
    plt.title("Dog" if np.argmax(predictions[i]) == 1 else "Cat")
    plt.axis('off')


# In[ ]:




