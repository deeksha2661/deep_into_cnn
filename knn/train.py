import numpy as np


def compute_distance(v1, v2):
	return np.sqrt(np.sum((v1-v2)**2, axis = 1, keepdims = True))

def k_nn_indices(X, eg, k):
	dist = compute_distance(X, eg)
	n = np.arange(0, X.shape[0], 1)
	n = n.reshape(len(n), 1)
	#print(n.shape, dist.shape)
	dis = np.append(n, dist, axis = 1)
	dis_sort = dis[dis[:, 1].argsort()]
	k_nn_index = dis_sort[0:k, 0:1]
	return k_nn_index

def classify_points(X, Y, x, k):
	y = []
	for eg in x:
		k_top = k_nn_indices(X, eg, k)
		k_top = k_top.reshape(k_top.shape[0])
		k_label = []
		#print(k_top)
		for i in k_top:
			#print(i)
			k_label.append(Y[int(i)])
        #print(k_label)
		most_frequent_label = max(set(k_label), key = k_label.count)
		y.append(most_frequent_label)
    
	return y
