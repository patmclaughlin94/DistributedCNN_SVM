import socket
import sys
import pickle
import numpy as np

# Test array
a = []
for i in range(9216):
	a.append(i)
print(sys.getsizeof(a))
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind the socket to a public host and a well known port
serversocket.bind(('localhost', 9000))
# become a server socket
serversocket.listen(5)

while True:
	# accept connections from outside
	(clientsocket, address) = serversocket.accept()
	# now do something with client socket
	serialized = pickle.dumps(a, protocol=0)
	clientsocket.send(serialized)
	clientsocket.close()
	
