import socket
import numpy as np
import pickle
#import keras
#from keras.datasets import mnist
import sys
from sklearn import linear_model
from sklearn.externals import joblib
import time

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# create socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clf = linear_model.SGDClassifier()
host = '127.0.0.1'
port = 25000

# connect to localhost
s.connect((host, port))
features = []
classifications = []
feature = []
classification = []
received_count = 0;
startTime = time.time()
num_batch = 0
while 1:
	print(classification)
	bad_feature = False
	bad_class = False
	feature_msg = s.recv(100000)
	if( not feature_msg): break
	try:
		feature = pickle.loads(feature_msg)
	except:
		bad_feature = True;
		# send message to server saying bad feature received
		# server will try to resend
		s.send(pickle.dumps('bf'))
	if(not bad_feature):
		# send message to server saying good featrue received
		s.send(pickle.dumps('gf'))
		class_msg = s.recv(100000)
		if(not class_msg): break
		try:
			print('trying classification!')
			classification = pickle.loads(class_msg)
		except:
			print('bad class!!!')
			bad_class = True;
			classification = []
			# send message to server saying bad classification received
			# server will try to resend			
			s.send(pickle.dumps('bc'))
		if(not bad_class):
			if(type(classification) == type([])):
				classification = []
				s.send(pickle.dumps('bc'))
			else:
				features.append(feature)
				classifications.append(classification)
				print('good class and good feature!', received_count)
				if(received_count > 99):
					# start training svm on 4999 samples
					print(len(features))
					print(len(classifications))
					clf.partial_fit(np.array(features), np.array(classifications), np.array([0,1,2,3,4,5,6,7,8,9]))
					num_batch = num_batch + 1
					features = []
					classifications = []
					received_count=0
				received_count=received_count+1
				s.send(pickle.dumps('gc'))

endTime = time.time()
runTime = endTime - startTime
print('Training 20,000 takes', runTime, ' seconds')

joblib.dump(clf, 'mnist_svm.pkl') 
s.close()
