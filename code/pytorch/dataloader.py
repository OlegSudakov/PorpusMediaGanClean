import numpy as np

def iterate(X, batch_size):
	np.random.shuffle(X)
	for i in range(X.shape[0] // batch_size):
		yield X[i*batch_size : (i+1)*batch_size,:,:,:]
	if (i+1)*batch_size < X.shape[0]:
		yield X[(i+1)*batch_size:,:,:,:]
