from collections import defaultdict
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np
from random import shuffle
import sys

def readData(filename):
	ratings_read = 0
	fi = open(filename, "r")
	entities = defaultdict()
	data = []
	for line in fi:
		a = line.split()
		if(len(a) > 1):
			e1 = a[0].strip();
			e2 = a[1].strip();
			rate = float(a[2].strip())
			if e1 not in entities.keys():
				entities[e1] = len(entities)
			if e2 not in entities.keys():
				entities[e2] = len(entities)
			ratings_read = ratings_read + 1;
			data.append([entities[e1], entities[e2], rate])	

		if(ratings_read % 1000 == 0):
			print ratings_read, " ratings read";		
	return	entities, data

def printData(data):
	for key in data:
		print key
		for key1 in data[key]:
			print "\t", key1, data[key][key1]	


def mae(data, predict):
	for d in data:
			mae = mae + abs(predict(d[0], d[1]) - d[2])
	mae = mae / len(data)

def mf(trdata, vdata, tdata, K, lr, reg, epoch):
	srng = RandomStreams(seed=234)
	rng = np.random
	rate_t = T.dscalar('rate_true')
	reg_con = T.dscalar('lambda')
	e1 = T.iscalar('e1')
	e2 = T.iscalar('e2')
	alpha = theano.shared(rng.normal(loc=0.0, scale=0.01), name="alpha") 
	b = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(entities))), name="bias")
	phi = theano.shared(rng.normal(loc=0.0, scale=0.01, size=(len(entities), K)), name="phi")
	
	prediction = alpha + b[e1] + b[e2] + T.dot(phi[e1], phi[e2])
	cost = (prediction - rate_t)*(prediction - rate_t) + reg_con*(alpha*alpha + b[e1]*b[e1] + b[e2]*b[e2] + T.dot(phi[e1],phi[e1]) + T.dot(phi[e2],phi[e2]) )

	#ga, gb1, gb2, gphi1, gphi2 = T.grad(cost, [alpha, b1, b2, phi1, phi2])
	ga, gb, gphi = T.grad(cost, [alpha, b, phi])
	
	predict = theano.function(inputs=[e1, e2], outputs = prediction)
	c = theano.function(inputs=[e1, e2, rate_t, theano.Param(reg_con, default=reg)], outputs=cost)
	train = theano.function(inputs=[e1, e2, rate_t, theano.Param(reg_con, default=reg)], outputs=[prediction, cost], updates= ( (b, b - lr*gb), (alpha, alpha - lr*ga), (phi, phi - lr*gphi)) )


	mae_val = 0.0
	mae_test = 0.0
	for i in range(0, epoch):
		shuffle(trdata);
		for d in trdata:
			train(d[0], d[1], d[2])
				
		#mae_val = mae(vdata, predict)
		#mae_test = mae(tdata, predict)

		for d in vdata:
			mae_val = mae_val + pow(predict(d[0], d[1]) - d[2], 2)
		mae_val = mae_val / len(vdata)
		for d in tdata:
			mae_test = mae_test + pow(predict(d[0], d[1]) - d[2], 2)
		mae_test = mae_test / len(tdata)

		print "Epoch : ", i, "validation mean sq. error : ", mae_val
		print "Epoch : ", i, "testdataa  mean sq. error : ", mae_test
		print 



if __name__ == "__main__":
	entities, data = readData(sys.argv[1])
	train_perc = 0.8
	val_perc = 0.1
	#print entities	
	#mf(entities, data, 5, 0.01, 0.01)
	shuffle(data);
	trlen = int(len(data)*train_perc)
	vallen = int(len(data)*val_perc)
	trdata = data[0:trlen]
	vdata = data[trlen:trlen+vallen]
	testdata = data[trlen+vallen:-1]
	
	print len(data)
	print len(trdata), len(vdata), len(testdata)

	mf(trdata, vdata, testdata, K=5, lr=0.01, reg=0.001, epoch=50)




			
