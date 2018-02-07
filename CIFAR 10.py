
# coding: utf-8

# In[1]:


import keras
import numpy as np
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print x_train.shape[1:] 
print x_train.shape[0]
y_test.shape
y_train=keras.utils.to_categorical(y_train,num_classes=10)
y_test=keras.utils.to_categorical(y_test,num_classes=10)
y_test.shape

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten,Activation


from keras.preprocessing.image import ImageDataGenerator

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=x_train.shape[1:],padding='same'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

opt=keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
x_train.astype('float32')
x_test.astype('float32')
x_train/=255
x_test/=255
model.fit(x_train,y_train,batch_size=128,epochs=100)

