import socket
import numpy as np
import pickle
#import keras
#from keras.datasets import mnist
import sys
from sklearn import linear_model
from sklearn.externals import joblib

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
# create socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clf = linear_model.SGDClassifier()
host = '10.0.0.5'
port = 25000

# connect to localhost
s.connect((host, port))
features = []
classifications = []
feature = []
classification = []
received_count = 0;

############
# Load SVM #
############
clf = joblib.load('mnist_svm.pkl')
num_samples = 0;
num_correct = 0;
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
				predicted_class = clf.predict(np.array([feature]))
				num_samples = num_samples+1
				print('predicted: ', predicted_class)
				print('actual: ', classification)
				if((predicted_class[0] - classification) < 0.1):
					num_correct = num_correct+1
				print('score: ', float(float(num_correct)/float(num_samples)))
				s.send(pickle.dumps('gc'))

#joblib.dump(clf, 'mnist_svm.pkl') 
s.close()
