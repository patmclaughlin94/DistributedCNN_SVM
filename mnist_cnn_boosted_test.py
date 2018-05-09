from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import socket
import pickle
import numpy as np
import sys
import time

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.summary()

# extract outputs from layer before dense layers
layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(x_train)

# Socket creation
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind socket to localhost
serversocket.bind(('', 25000))

# become a server socket
serversocket.listen(500)
while True:
	# accept connections from outside
	(clientsocket, address) = serversocket.accept()
	# now do something with client socket
	retryFlg = False
	# Streaming: 
	for i in range(5000):
		if(retryFlg == True):
			i = i - 1
			print('im going to reset!!!')
			retryFlg = False
		x_feature = intermediate_layer_model.predict((x_test[i]).reshape(1,28,28,1))
		classification = y_test[i]
		# transmit
		serialized_feature = pickle.dumps((x_feature[0]).tolist())
		## TEST REMOVE PLEASE!!! ##
		#test = pickle.dumps((np.array([1.0, 2.0, 0.002])).tolist())
		serialized_class = pickle.dumps(classification)
		clientsocket.send(serialized_feature)
		#clientsocket.send(test)
		print('Ive sent my serialized feature of size', sys.getsizeof(serialized_feature))
		msg = clientsocket.recv(1024)
		print('Ive received a message')
		clientMessage = pickle.loads(msg)
		print('Ive loaded this message: ', type(clientMessage))
		if(clientMessage == 'bf'):
			print('BAD')
  			retryFlg = True
  		else:
			print(classification)
  			clientsocket.send(serialized_class)
  			msg = clientsocket.recv(1024)
  			clientMessage = pickle.loads(msg)
  			if(clientMessage == 'bc'):
  				retryFlg = True
			
  	#while(True):
  	#	print('Im bad every single time')
  	#	retryFlg = True
  	#else:
  	#	print(' Im in the else box!!!')
  	#	print(classification)
  	#	serialized_class = pickle.dumps(classification)
  	#	clientsocket.send(serialized_class)
  	#	msg = clientsocket.recv(1024)
  	#	clientMessage = pickle.loads(msg)
  	#	if(clientMessage == 'bc'):
  	#		retryFlg = True
	
	print('closing the client socket...')
	clientsocket.close()
	break;
print('closing the server socket...')
serversocket.close()
# Save feature outputs to text file
#np.savetxt('intermediate_output.txt',intermediate_output)
#np.savetxt('expected_output.txt', y_train)
# In order to pop the last classification layer:
#	loaded_model.layers.pop()
#loaded_model.compile(loss=keras.losses.categorical_crossentropy,
 #             optimizer=keras.optimizers.Adadelta(),
 #             metrics=['accuracy'])

#score = loaded_model.evaluate(x_test, y_test, verbose=0)


#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
