from keras.models import  Sequential # To build nerual network
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout # to creat layers
from keras.utils import to_categorical # To convet matrix array into o's ans 1's
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') # style of matplotlib

from keras.datasets import cifar10

(xtrain,ytrain),(xtest,ytest) = cifar10.load_data()

#img = plt.imshow(xtrain[4])

#lab = ytrain[10]
#print(lab)

classification = ['airplane','automobile','bird' ,'cat','deer','dog','frog','horse','ship','truck']

#print('The image is : ', classification[ytrain[4][0]])
#xtest.shape
#ytest.shape
#xtrain.shape
#ytrain.shape

ytrain_one_hot = to_categorical(ytrain)
ytest_one_hot = to_categorical(ytest)

ytest_one_hot[100]

xtrain = xtrain/255
xtest = xtest/255

model = Sequential()
model.add(Conv2D(32,(5,5) , activation="relu",input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32,(5,5) , activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1000, activation ="relu"))
model.add(Dropout(0.5))

model.add(Dense(250 , activation ="relu"))

model.add(Dense(10, activation= "softmax"))

#model.summary()

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

tr = model.fit(xtrain , ytrain_one_hot,batch_size=256,epochs=15,validation_split=0.2)

model.evaluate(xtest,ytest_one_hot)

image = plt.imread('')   # insert image url here
from skimage import transform
resize = transform.resize(image,(32,32,3))
prediction = model.predict(np.expand_dims(resize, axis=0))

list_index = [0,1,2,3,4,5,6,7,8,9]
x = prediction
for i in range(10):
  for j in range(10):
    if x[0] [list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp
print(list_index)

for i in range(1):
  print("This is a",classification[list_index[i]] ,"image")