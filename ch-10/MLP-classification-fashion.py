import tensorflow as tf
from tensorflow import keras 

print(tf.__version__)
print(keras.__version__)

#load keras fashion dataset (includes 70k grayscale images 28x28p each with 10 classes)
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()#every image is a 28x28 array, pixel intensities are integers 0->255

#show the data type of the training set
print(X_train_full.shape)#(60k, 28, 28)
print(X_train_full.dtype)#uint8

#since we will train the neural net using gradient descent we must scale the i/p features
#we'll scaled down to 0-1 range by simply dividing by 255.0 (also converting them to floats)
#also while we are at it we will create the validation set
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, ytrain = y_train_full[:5000],y_train_full[5000:]

#to map the o/p nums
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#e.g.
print(class_names[ytrain[0]])#'coat'

#building the classification MLP
model = keras.models.Sequential()#simplest keras model for a neural net. a stack of layers connected sequentially
model.add(keras.layers.Flatten(input_shape=[28,28]))#role is to convert each i/p img into a 1d arr. input shape is the shape of an input instance
model.add(keras.layers.Dense(300, activation="relu"))#dense hiddel layer with 300 neurons and using relu activation function
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))#dense o/p layer with 10 neurons (1 per class) using softmax activation function (since they are exclusive)

model.summary()#displays all the model's layers 
#loss: we use sparse_categorical_crossentropy because we have sparse labels (for each instance there is a target class index 0-9) and the classes are exclusive
# if instead we have one target probability per class for each instance (e.g. [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class three)
# then we would've needed to use "categorical_crossentropy"
# if we were doing binary classification we would use sigmoid activation in the op layer and loss "binary_crossentropy"

#optimizer: sgd => stochastic gradient descent i.e. keras will perform the backpropagation algo
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


#here we use the optional validation set which is very helpful
#for example if the perf on the training set is much better than the validation set the model is probably overfitting the training set

#alternatively much easier we could have just did validation_split=0.1 (use 10% of training data for validation)
history = model.fit(X_train, ytrain, epochs=30, validation_data=(X_valid, y_valid))

#If the training set was very skewed, with some classes being overrepresented and oth‐
#ers underrepresented, it would be useful to set the class_weight argument when
#calling the fit() method, which would give a larger weight to underrepresented
#lasses and a lower weight to overrepresented classes. These weights would be used by
#Keras when computing the loss.


#plot the training and validation accuracies
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


#test the model on the test set
model.evaluate(X_test, y_test)

#using the model to make predictions
#we don't have actual new ones so we'll reuse the first three on the test set
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
print(y_proba)



