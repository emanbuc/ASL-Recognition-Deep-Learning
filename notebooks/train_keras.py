import argparse
import json
import os
from azureml.core import Run
from azureml.core.model import Model
import pickle
import keras
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


parser = argparse.ArgumentParser(description='MNIST Train')
parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden', type=int, default=100)
parser.add_argument('--dropout', type=float)

args = parser.parse_args()

mnist_fn = 'dataset/mnist.pkl' if args.data_folder is None else os.path.join(args.data_folder, 'sign-language-mnist-data','sign-language-mnist.pkl')
with open(mnist_fn,'rb') as f:
    X_train,X_test,y_train,y_test = pickle.load(f)

input_shape = (28,28, 1) # 28*28 = 784
# Creating a Sequential Model and adding the layers
model = keras.Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(args.hidden, activation=tf.nn.relu))
if args.dropout is not None and args.dropout<1:
    model.add(Dropout(args.dropout))
model.add(Dense(y_train.shape[1],activation=tf.nn.softmax))


model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

os.makedirs('outputs',exist_ok=True)
model.save('outputs/mnist_model.hdf5')

# Log metrics
try:
    run = Run.get_context()
    run.log('Test Loss', score[0])
    run.log('Accuracy', score[1])
except:
    print("Running locally")